import os
import numpy as np
import time
import pickle
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from imblearn.metrics import geometric_mean_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import TensorDataset, DataLoader
from spikebench import load_retina
from load_retina_all import load_retina_all
from tsai.models.InceptionTimePlus import InceptionTimePlus
from tsai.models.FCNPlus import FCNPlus
from tsai.models.ResNetPlus import ResNetPlus
from tsai.models.XceptionTimePlus import XceptionTimePlus
from modified_models import ModifiedInceptionTimePlus, ModifiedFCNPlus, ModifiedResNetPlus, ModifiedXceptionTimePlus

def sample_distance_feature_from_agg(X, bins, agg=None, signed=False):
    if agg is None:
        agg = np.array(X.reshape(-1))
    else:
        agg = np.array(agg)
    agg.sort()
    Xtemp = np.array(X)
    Xtemp.sort(axis=1)
    dim = X.shape[1]
    sdf = np.zeros((len(X), bins))
    for i in range(len(X)):
        # print(agg)
        # print(np.repeat(Xtemp[i], len(agg)//dim))
        if signed:
            sdf[i] = (np.repeat(Xtemp[i], len(agg)//dim) - agg).reshape(bins, -1).mean(axis=-1)
        else:
            sdf[i] = np.abs(agg - np.repeat(Xtemp[i], len(agg)//dim)).reshape(bins, -1).mean(axis=-1)
    return sdf

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_layers=2):
        super(MLP, self).__init__()
        layerlist = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for i in range(hidden_layers-1):
            layerlist.append(nn.Linear(hidden_dim, hidden_dim))
            layerlist.append(nn.ReLU())
        layerlist.append(nn.Linear(hidden_dim, output_dim))
        
        self.model = nn.Sequential(*layerlist)
    
    def forward(self, x):
        return self.model(x)

# Define the 1D CNN model
class Simple1DCNN(nn.Module):
    def __init__(self, sequence_length, input_channels = 1, num_outputs = 2):
        super(Simple1DCNN, self).__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, stride=1, padding=1)  # Conv layer
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)  # Conv layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # Max pooling layer
        self.fc1 = nn.Linear(32 * (sequence_length//4), 64)  # Fully connected layer
        self.fc2 = nn.Linear(64, num_outputs)  # Output layer

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.view(x.shape[0], 1, x.shape[1])
        x = torch.relu(self.conv1(x))  # Apply first conv + ReLU
        x = self.pool(x)  # Apply pooling
        x = torch.relu(self.conv2(x))  # Apply second conv + ReLU
        x = self.pool(x)  # Apply pooling
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))  # Apply fully connected layer
        x = self.fc2(x)  # Output layer
        return x

class BatchStandardize:
    def __call__(self, batch):
        if len(batch.shape) == 3:
            mean = batch.mean(dim=(0, 2), keepdim=True)  # Mean over batch and time dimensions
            std = batch.std(dim=(0, 2), keepdim=True)   # Std over batch and time dimensions
        else:
            mean = batch.mean(dim=(0, 1), keepdim=True)  # Mean over batch and time dimensions
            std = batch.std(dim=(0, 1), keepdim=True)   # Std over batch and time dimensions
        std[std == 0] = 1  # Avoid division by zero
        return (batch - mean) / (std+1e-6)
        
def get_balanced_data(X, y, size_per_class=None):
    # Ensure labels are integers for indexing
    unique_classes = torch.unique(y)
    N_cls = min([(y == cls).sum().item() for cls in unique_classes])
    if size_per_class is None or N_cls < size_per_class:
        size_per_class = N_cls
    # print(N_cls)
        
    indices = []
    for cls in unique_classes:
        # Find indices of the current class
        class_indices = torch.where(y == cls)[0]
        
        # Randomly select `size_per_class` samples from the class
        selected_indices = class_indices[torch.randperm(len(class_indices))[:size_per_class]]
        indices.append(selected_indices)
    
    # Concatenate indices for all classes
    indices = torch.cat(indices)
    
    # Extract test data and labels
    X = X[indices]
    y = y[indices]
    
    return X, y

# Testing code for the case using only the ISI feature
def test_model(model, test_loader, criterion, batch_standardize, device, num_outputs = 2):
    model.eval()  # Set the model to evaluation mode
    curloss = 0
    correct = 0
    total = 0
    ys = []
    preds = []
    pred_probs = []
    if batch_standardize:
        batch_standardizer = BatchStandardize()
    with torch.no_grad():  # Disable gradient computation for testing
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
            if batch_standardize:
                inputs = batch_standardizer(inputs)
            outputs = model(inputs)  # Forward pass
            total += labels.size(0)
            loss = criterion(outputs, labels)
            curloss += loss.item()*len(labels)

            # Convert logits to predictions
            pred = torch.argmax(outputs, dim=1)       # Predicted classes
            if num_outputs == 2:
                pred_prob = torch.softmax(outputs, dim=1)[:, 1]  # Probabilities for class 1
            else:
                pred_prob = torch.softmax(outputs, dim=1)

            correct += (pred == labels).sum().item()
            ys.append(labels.cpu().numpy())
            preds.append(pred.detach().cpu().numpy())
            pred_probs.append(pred_prob.detach().cpu().numpy())
        
    loss = curloss / total
    acc = correct / total
    ys = np.concatenate(ys)
    preds = np.concatenate(preds)
    pred_probs = np.concatenate(pred_probs)
    kappa = cohen_kappa_score(ys, preds)
    auc2 = 0
    gmean2 = 0
    if num_outputs == 2:
        auc = roc_auc_score(ys, pred_probs)
        gmean = geometric_mean_score(ys.ravel(), preds.ravel())
    else:
        auc = roc_auc_score(ys, pred_probs, multi_class='ovr', average='weighted')
        auc2 = roc_auc_score(ys, pred_probs, multi_class='ovr', average='macro')
        gmean = geometric_mean_score(ys.ravel(), preds.ravel(), average='weighted')
        gmean2 = geometric_mean_score(ys.ravel(), preds.ravel(), average='macro')
    # print(ys.shape, preds.shape)
    return loss, acc, kappa, auc, auc2, gmean, gmean2

# Testing code for the case using the ISI feature and additional features
def test_model2(model, test_loader, criterion, batch_standardize, device, num_outputs = 2):
    model.eval()  # Set the model to evaluation mode
    curloss = 0
    correct = 0
    total = 0
    ys = []
    preds = []
    pred_probs = []
    if batch_standardize:
        batch_standardizer = BatchStandardize()
    with torch.no_grad():  # Disable gradient computation for testing
        for inputs, inputs2, labels in test_loader:
            inputs, inputs2, labels = inputs.to(device), inputs2.to(device), labels.to(device)  # Move data to device
            if batch_standardize:
                inputs = batch_standardizer(inputs)
                inputs2 = batch_standardizer(inputs2)
            outputs = model(inputs, inputs2)  # Forward pass
            total += labels.size(0)
            loss = criterion(outputs, labels)
            curloss += loss.item()*len(labels)

            # Convert logits to predictions
            pred = torch.argmax(outputs, dim=1)       # Predicted classes
            if num_outputs == 2:
                pred_prob = torch.softmax(outputs, dim=1)[:, 1]  # Probabilities for class 1
            else:
                pred_prob = torch.softmax(outputs, dim=1)

            correct += (pred == labels).sum().item()
            ys.append(labels.cpu().numpy())
            preds.append(pred.detach().cpu().numpy())
            pred_probs.append(pred_prob.detach().cpu().numpy())
        
    loss = curloss / total
    acc = correct / total
    ys = np.concatenate(ys)
    preds = np.concatenate(preds)
    pred_probs = np.concatenate(pred_probs)
    kappa = cohen_kappa_score(ys, preds)
    auc2 = 0
    gmean2 = 0
    if num_outputs == 2:
        auc = roc_auc_score(ys, pred_probs)
        gmean = geometric_mean_score(ys.ravel(), preds.ravel())
    else:
        auc = roc_auc_score(ys, pred_probs, multi_class='ovr', average='weighted')
        auc2 = roc_auc_score(ys, pred_probs, multi_class='ovr', average='macro')
        gmean = geometric_mean_score(ys.ravel(), preds.ravel(), average='weighted')
        gmean2 = geometric_mean_score(ys.ravel(), preds.ravel(), average='macro')
    # print(ys.shape, preds.shape)
    return loss, acc, kappa, auc, auc2, gmean, gmean2
    
# Define the learning rate schedule function
def flat_cos_schedule(step, pct_start, T_max):
    if step < pct_start * T_max:
        return 1.0  # Flat phase
    else:
        # Cosine annealing phase
        cos_progress = (step - pct_start * T_max) / (1 - pct_start) / T_max
        return 0.5 * (1 + torch.cos(torch.tensor(cos_progress * torch.pi)))

# Training code for the case using only the ISI feature
def train_cls(X_train, y_train, X_test, y_test, device, X_val=None, y_val=None, 
              use_balancing=False, 
              model_type = 'MLP', n_epochs = 100, batch_size = 32, 
              lr=0.001, wd=0, alg = 'Adam', schedule=None, batch_standardize=None, pct_start=0.75):
    use_val = False
    num_outputs = len(np.unique(y_train))
    # Convert to PyTorch tensors
    X_train_ = torch.tensor(X_train, dtype=torch.float32)
    y_train_ = torch.tensor(y_train, dtype=torch.long).view(-1,)
    X_test_ = torch.tensor(X_test, dtype=torch.float32)
    y_test_ = torch.tensor(y_test, dtype=torch.long).view(-1,)
    if X_val is not None and y_val is not None:
        X_val_ = torch.tensor(X_val, dtype=torch.float32)
        y_val_ = torch.tensor(y_val, dtype=torch.long).view(-1,)
        use_val = True
    # print('start to build model')
    # Initialize the model, loss function, and optimizer
    if model_type == 'MLP':
        input_dim = X_train.shape[1]
        hidden_dim = 64
        output_dim = num_outputs
        hidden_layers = 2
        model = MLP(input_dim, hidden_dim, output_dim, hidden_layers).to(device)
        # print('building MLP')
    elif model_type == '1dCNN':
        if len(X_train.shape) == 2:
            model = Simple1DCNN(X_train.shape[1], input_channels = 1, num_outputs = num_outputs).to(device)
        elif len(X_train.shape) == 3:
            model = Simple1DCNN(X_train.shape[2], input_channels = X_train.shape[1], num_outputs = num_outputs).to(device)
        # print('building 1dCNN')
    elif model_type == 'InceptionTimePlus':
        model = InceptionTimePlus(X_train.shape[1], num_outputs).to(device)
    elif model_type == 'FCNPlus':
        model = FCNPlus(X_train.shape[1], num_outputs).to(device)
    elif model_type == 'ResNetPlus':
        model = ResNetPlus(X_train.shape[1], num_outputs).to(device)
    elif model_type == 'XceptionTimePlus':
        model = XceptionTimePlus(X_train.shape[1], num_outputs).to(device)
    else:
        raise NotImplementedError
        
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    if alg == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif alg == 'SGD':
        print('use SGD')
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise NotImplementedError
    if schedule is not None:
        if schedule=='flatcosine':
            print('use flatcosine scheduler')
            # Define the total number of steps
            if use_balancing:
                unique_classes = torch.unique(y_train_)
                N_cls = min([(y_train_ == cls).sum().item() for cls in unique_classes])
                T_max = n_epochs * ((2 * N_cls - 1) // batch_size + 1)
            else:
                T_max = n_epochs * ((len(X_train) - 1) // batch_size + 1)
            # pct_start = 0.75  # Fraction of steps for the flat phase
            # print(f'Tmax: {T_max}, flat for {int(pct_start * T_max)} and cosine for {int((1 - pct_start) * T_max)}')
            
            # Apply the schedule with LambdaLR
            scheduler = LambdaLR(optimizer, lr_lambda=lambda step: flat_cos_schedule(step, pct_start, T_max))
        elif schedule=='cosine':
            print('use cosine scheduler')
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.01*lr)
        
    if batch_standardize:
        print('use batch standardize')
        # Custom batch transformation
        batch_standardizer = BatchStandardize()

    loss_traj = []
    acc_traj = []
    kappa_traj = []
    auc_traj = []
    auc2_traj = []
    gmean_traj = []
    gmean2_traj = []
    valloss_traj = []
    valacc_traj = []
    valkappa_traj = []
    valauc_traj = []
    valauc2_traj = []
    valgmean_traj = []
    valgmean2_traj = []
    testloss_traj = []
    testacc_traj = []
    testkappa_traj = []
    testauc_traj = []
    testauc2_traj = []
    testgmean_traj = []
    testgmean2_traj = []
    lr_traj = []
    
    # Training loop
    X_train, y_train = X_train_, y_train_
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    X_test, y_test = X_test_, y_test_
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    if use_val:
        X_val, y_val = X_val_, y_val_
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    for epoch in range(n_epochs):
        if use_balancing:
            X_train, y_train = get_balanced_data(X_train_, y_train_)
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            print(len(train_loader))
            X_test, y_test = get_balanced_data(X_test_, y_test_)
            test_dataset = TensorDataset(X_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
            if use_val:
                X_val, y_val = get_balanced_data(X_val_, y_val_)
                val_dataset = TensorDataset(X_val, y_val)
                val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
            
        model.train()
        N, curloss, correct = 0, 0, 0
        ys = []
        preds = []
        pred_probs = []
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            if batch_standardize:
                batch_X = batch_standardizer(batch_X)
                
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
    
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if schedule is not None:
                scheduler.step()
                lr_traj.append(scheduler.get_last_lr()[0])
            N += len(batch_X)
            curloss += loss.item()*len(batch_X)

            # Convert logits to predictions
            pred = torch.argmax(outputs, dim=1)       # Predicted classes
            if num_outputs == 2:
                pred_prob = torch.softmax(outputs, dim=1)[:, 1]  # Probabilities for class 1
            else:
                pred_prob = torch.softmax(outputs, dim=1)
            correct += (pred == batch_y).sum().item()
            ys.append(batch_y.cpu().numpy())
            preds.append(pred.detach().cpu().numpy())
            pred_probs.append(pred_prob.detach().cpu().numpy())
        loss_traj.append(curloss/N)
        acc_traj.append(correct/N)
        ys = np.concatenate(ys)
        preds = np.concatenate(preds)
        pred_probs = np.concatenate(pred_probs)
        kappa_traj.append(cohen_kappa_score(ys, preds))
        if num_outputs == 2:
            auc_traj.append(roc_auc_score(ys, pred_probs))
            auc2_traj.append(0)
            gmean_traj.append(geometric_mean_score(ys.ravel(), preds.ravel()))
            gmean2_traj.append(0)
        else:
            auc_traj.append(roc_auc_score(ys, pred_probs, multi_class='ovr', average='weighted'))
            auc2_traj.append(roc_auc_score(ys, pred_probs, multi_class='ovr', average='macro'))
            gmean_traj.append(geometric_mean_score(ys.ravel(), preds.ravel(), average='weighted'))
            gmean2_traj.append(geometric_mean_score(ys.ravel(), preds.ravel(), average='macro'))
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {curloss/N:.4f}, ACC: {correct/N:.4f}, kappa: {kappa_traj[-1]:.4f}, AUC: {auc_traj[-1]:.4f}, AUC2: {auc2_traj[-1]:.4f}, gmean: {gmean_traj[-1]:.4f}, gmean2: {gmean2_traj[-1]:.4f}")
    
        # Evaluation
        testloss, testacc, testkappa, testauc, testauc2, testgmean, testgmean2 \
        = test_model(model, test_loader, criterion, batch_standardize, device, num_outputs)
        testloss_traj.append(testloss)
        testacc_traj.append(testacc)
        testkappa_traj.append(testkappa)
        testauc_traj.append(testauc)
        testauc2_traj.append(testauc2)
        testgmean_traj.append(testgmean)
        testgmean2_traj.append(testgmean2)
        if epoch == n_epochs - 1:
            print(f"Test Loss: {testloss_traj[-1]:.4f}, Test ACC: {testacc_traj[-1]:.4f}, Test kappa: {testkappa_traj[-1]:.4f}, Test AUC: {testauc_traj[-1]:.4f}, Test AUC2: {testauc2_traj[-1]:.4f}, Test gmean: {testgmean_traj[-1]:.4f}, Test gmean2: {testgmean2_traj[-1]:.4f}")
        if use_val:
            valloss, valacc, valkappa, valauc, valauc2, valgmean, valgmean2 \
            = test_model(model, val_loader, criterion, batch_standardize, device, num_outputs)
            valloss_traj.append(valloss)
            valacc_traj.append(valacc)
            valkappa_traj.append(valkappa)
            valauc_traj.append(valauc)
            valauc2_traj.append(valauc2)
            valgmean_traj.append(valgmean)
            valgmean2_traj.append(valgmean2)
            if epoch == n_epochs - 1:
                print(f"Val Loss: {valloss_traj[-1]:.4f}, Val ACC: {valacc_traj[-1]:.4f}, Val kappa: {valkappa_traj[-1]:.4f}, Val AUC: {valauc_traj[-1]:.4f}, Val AUC2: {valauc2_traj[-1]:.4f}, Val gmean: {valgmean_traj[-1]:.4f}, Val gmean2: {valgmean2_traj[-1]:.4f}")

    loss_traj, valloss_traj, testloss_traj = np.array(loss_traj), np.array(valloss_traj), np.array(testloss_traj)
    acc_traj, valacc_traj, testacc_traj = np.array(acc_traj), np.array(valacc_traj), np.array(testacc_traj)
    kappa_traj, valkappa_traj, testkappa_traj = np.array(kappa_traj), np.array(valkappa_traj), np.array(testkappa_traj)
    auc_traj, valauc_traj, testauc_traj = np.array(auc_traj), np.array(valauc_traj), np.array(testauc_traj)
    auc2_traj, valauc2_traj, testauc2_traj = np.array(auc2_traj), np.array(valauc2_traj), np.array(testauc2_traj)
    gmean_traj, valgmean_traj, testgmean_traj = np.array(gmean_traj), np.array(valgmean_traj), np.array(testgmean_traj)
    gmean2_traj, valgmean2_traj, testgmean2_traj = np.array(gmean2_traj), np.array(valgmean2_traj), np.array(testgmean2_traj)
    results = {}
    results['loss'] = loss_traj
    results['acc'] = acc_traj
    results['kappa'] = kappa_traj
    results['auc'] = auc_traj
    results['auc2'] = auc2_traj
    results['gmean'] = gmean_traj
    results['gmean2'] = gmean2_traj
    results['valloss'] = valloss_traj
    results['valacc'] = valacc_traj
    results['valkappa'] = valkappa_traj
    results['valauc'] = valauc_traj
    results['valauc2'] = valauc2_traj
    results['valgmean'] = valgmean_traj
    results['valgmean2'] = valgmean2_traj
    results['testloss'] = testloss_traj
    results['testacc'] = testacc_traj
    results['testkappa'] = testkappa_traj
    results['testauc'] = testauc_traj
    results['testauc2'] = testauc2_traj
    results['testgmean'] = testgmean_traj
    results['testgmean2'] = testgmean2_traj
    results['lr_traj'] = lr_traj
    
    return model, results

# Training code for the case using the ISI feature and additional features
def train_cls2(X_train, X_train2, y_train, X_test, X_test2, y_test, device, X_val=None, X_val2=None, y_val=None, 
              use_balancing=False, 
              model_type = 'MLP', n_epochs = 100, batch_size = 32, 
              lr=0.001, wd=0, alg = 'Adam', schedule=None, batch_standardize=None, use_CNN_for_additional=False, pct_start=0.75):
    use_val = False
    num_outputs = len(np.unique(y_train))
    # Convert to PyTorch tensors
    X_train_ = torch.tensor(X_train, dtype=torch.float32)
    X_train2_ = torch.tensor(X_train2, dtype=torch.float32)
    y_train_ = torch.tensor(y_train, dtype=torch.long).view(-1,)
    X_test_ = torch.tensor(X_test, dtype=torch.float32)
    X_test2_ = torch.tensor(X_test2, dtype=torch.float32)
    y_test_ = torch.tensor(y_test, dtype=torch.long).view(-1,)
    
    if X_val is not None and y_val is not None and X_val2 is not None:
        X_val_ = torch.tensor(X_val, dtype=torch.float32)
        X_val2_ = torch.tensor(X_val2, dtype=torch.float32)
        y_val_ = torch.tensor(y_val, dtype=torch.long).view(-1,)
        use_val = True
        
    # print('start to build model')
    # Initialize the model, loss function, and optimizer
    if model_type == 'InceptionTimePlus':
        model = ModifiedInceptionTimePlus(X_train.shape[1], num_outputs, X_train2.shape[1], X_train2.shape[2], use_CNN_for_additional).to(device)
    elif model_type == 'FCNPlus':
        model = ModifiedFCNPlus(X_train.shape[1], num_outputs, X_train2.shape[1], X_train2.shape[2], use_CNN_for_additional).to(device)
    elif model_type == 'ResNetPlus':
        model = ModifiedResNetPlus(X_train.shape[1], num_outputs, X_train2.shape[1], X_train2.shape[2], use_CNN_for_additional).to(device)
    elif model_type == 'XceptionTimePlus':
        model = ModifiedXceptionTimePlus(X_train.shape[1], num_outputs, X_train2.shape[1], X_train2.shape[2], use_CNN_for_additional).to(device)
    else:
        raise NotImplementedError
        
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    if alg == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif alg == 'SGD':
        print('use SGD')
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise NotImplementedError
    if schedule is not None:
        if schedule=='flatcosine':
            print('use flatcosine scheduler')
            # Define the total number of steps
            if use_balancing:
                unique_classes = torch.unique(y_train_)
                N_cls = min([(y_train_ == cls).sum().item() for cls in unique_classes])
                T_max = n_epochs * ((2 * N_cls - 1) // batch_size + 1)
            else:
                T_max = n_epochs * ((len(X_train) - 1) // batch_size + 1)
            # pct_start = 0.75  # Fraction of steps for the flat phase
            # print(f'Tmax: {T_max}, flat for {int(pct_start * T_max)} and cosine for {int((1 - pct_start) * T_max)}')
            
            # Apply the schedule with LambdaLR
            scheduler = LambdaLR(optimizer, lr_lambda=lambda step: flat_cos_schedule(step, pct_start, T_max))
        elif schedule=='cosine':
            print('use cosine scheduler')
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.01*lr)
        
    if batch_standardize:
        # Custom batch transformation
        batch_standardizer = BatchStandardize()

    loss_traj = []
    acc_traj = []
    kappa_traj = []
    auc_traj = []
    auc2_traj = []
    gmean_traj = []
    gmean2_traj = []
    valloss_traj = []
    valacc_traj = []
    valkappa_traj = []
    valauc_traj = []
    valauc2_traj = []
    valgmean_traj = []
    valgmean2_traj = []
    testloss_traj = []
    testacc_traj = []
    testkappa_traj = []
    testauc_traj = []
    testauc2_traj = []
    testgmean_traj = []
    testgmean2_traj = []
    lr_traj = []
    
    # Training loop
    X_train, X_train2, y_train = X_train_, X_train2_, y_train_
    train_dataset = TensorDataset(X_train, X_train2, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    X_test, X_test2, y_test = X_test_, X_test2_, y_test_
    test_dataset = TensorDataset(X_test, X_test2, y_test)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    if use_val:
        X_val, X_val2, y_val = X_val_, X_val2_, y_val_
        val_dataset = TensorDataset(X_val, X_val2, y_val)
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    for epoch in range(n_epochs):
        if use_balancing:
            X_train, X_train2, y_train = get_balanced_data(X_train_, y_train_, X2=X_train2_)
            train_dataset = TensorDataset(X_train, X_train2, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            print(len(train_loader))
            X_test, X_test2, y_test = get_balanced_data(X_test_, y_test_, X2=X_test2_)
            test_dataset = TensorDataset(X_test, X_test2, y_test)
            test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
            if use_val:
                X_val, X_val2, y_val = get_balanced_data(X_val_, y_val_, X2=X_val2_)
                val_dataset = TensorDataset(X_val, X_val2, y_val)
                val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
            
        model.train()
        N, curloss, correct = 0, 0, 0
        ys = []
        preds = []
        pred_probs = []
        for batch_X, batch_X2, batch_y in train_loader:
            batch_X, batch_X2, batch_y = batch_X.to(device), batch_X2.to(device), batch_y.to(device)

            if batch_standardize:
                batch_X = batch_standardizer(batch_X)
                batch_X2 = batch_standardizer(batch_X2)
                
            # Forward pass
            outputs = model(batch_X, batch_X2)
            loss = criterion(outputs, batch_y)
    
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if schedule is not None:
                scheduler.step()
                lr_traj.append(scheduler.get_last_lr()[0])
            N += len(batch_X)
            curloss += loss.item()*len(batch_X)

            # Convert logits to predictions
            pred = torch.argmax(outputs, dim=1)       # Predicted classes
            if num_outputs == 2:
                pred_prob = torch.softmax(outputs, dim=1)[:, 1]  # Probabilities for class 1
            else:
                pred_prob = torch.softmax(outputs, dim=1)

            correct += (pred == batch_y).sum().item()
            ys.append(batch_y.cpu().numpy())
            preds.append(pred.detach().cpu().numpy())
            pred_probs.append(pred_prob.detach().cpu().numpy())
        loss_traj.append(curloss/N)
        acc_traj.append(correct/N)
        ys = np.concatenate(ys)
        preds = np.concatenate(preds)
        pred_probs = np.concatenate(pred_probs)
        kappa_traj.append(cohen_kappa_score(ys, preds))
        if num_outputs == 2:
            auc_traj.append(roc_auc_score(ys, pred_probs))
            auc2_traj.append(0)
            gmean_traj.append(geometric_mean_score(ys.ravel(), preds.ravel()))
            gmean2_traj.append(0)
        else:
            auc_traj.append(roc_auc_score(ys, pred_probs, multi_class='ovr', average='weighted'))
            auc2_traj.append(roc_auc_score(ys, pred_probs, multi_class='ovr', average='macro'))
            gmean_traj.append(geometric_mean_score(ys.ravel(), preds.ravel(), average='weighted'))
            gmean2_traj.append(geometric_mean_score(ys.ravel(), preds.ravel(), average='macro'))
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {curloss/N:.4f}, ACC: {correct/N:.4f}, kappa: {kappa_traj[-1]:.4f}, AUC: {auc_traj[-1]:.4f}, AUC2: {auc2_traj[-1]:.4f}, gmean: {gmean_traj[-1]:.4f}, gmean2: {gmean2_traj[-1]:.4f}")
    
        # Evaluation
        testloss, testacc, testkappa, testauc, testauc2, testgmean, testgmean2 \
        = test_model2(model, test_loader, criterion, batch_standardize, device, num_outputs)
        testloss_traj.append(testloss)
        testacc_traj.append(testacc)
        testkappa_traj.append(testkappa)
        testauc_traj.append(testauc)
        testauc2_traj.append(testauc2)
        testgmean_traj.append(testgmean)
        testgmean2_traj.append(testgmean2)
        if epoch == n_epochs - 1:
            print(f"Test Loss: {testloss_traj[-1]:.4f}, Test ACC: {testacc_traj[-1]:.4f}, Test kappa: {testkappa_traj[-1]:.4f}, Test AUC: {testauc_traj[-1]:.4f}, Test AUC2: {testauc2_traj[-1]:.4f}, Test gmean: {testgmean_traj[-1]:.4f}, Test gmean2: {testgmean2_traj[-1]:.4f}")
        if use_val:
            valloss, valacc, valkappa, valauc, valauc2, valgmean, valgmean2 \
            = test_model2(model, val_loader, criterion, batch_standardize, device, num_outputs)
            valloss_traj.append(valloss)
            valacc_traj.append(valacc)
            valkappa_traj.append(valkappa)
            valauc_traj.append(valauc)
            valauc2_traj.append(valauc2)
            valgmean_traj.append(valgmean)
            valgmean2_traj.append(valgmean2)
            if epoch == n_epochs - 1:
                print(f"Val Loss: {valloss_traj[-1]:.4f}, Val ACC: {valacc_traj[-1]:.4f}, Val kappa: {valkappa_traj[-1]:.4f}, Val AUC: {valauc_traj[-1]:.4f}, Val AUC2: {valauc2_traj[-1]:.4f}, Val gmean: {valgmean_traj[-1]:.4f}, Val gmean2: {valgmean2_traj[-1]:.4f}")

    loss_traj, valloss_traj, testloss_traj = np.array(loss_traj), np.array(valloss_traj), np.array(testloss_traj)
    acc_traj, valacc_traj, testacc_traj = np.array(acc_traj), np.array(valacc_traj), np.array(testacc_traj)
    kappa_traj, valkappa_traj, testkappa_traj = np.array(kappa_traj), np.array(valkappa_traj), np.array(testkappa_traj)
    auc_traj, valauc_traj, testauc_traj = np.array(auc_traj), np.array(valauc_traj), np.array(testauc_traj)
    auc2_traj, valauc2_traj, testauc2_traj = np.array(auc2_traj), np.array(valauc2_traj), np.array(testauc2_traj)
    gmean_traj, valgmean_traj, testgmean_traj = np.array(gmean_traj), np.array(valgmean_traj), np.array(testgmean_traj)
    gmean2_traj, valgmean2_traj, testgmean2_traj = np.array(gmean2_traj), np.array(valgmean2_traj), np.array(testgmean2_traj)
    results = {}
    results['loss'] = loss_traj
    results['acc'] = acc_traj
    results['kappa'] = kappa_traj
    results['auc'] = auc_traj
    results['auc2'] = auc2_traj
    results['gmean'] = gmean_traj
    results['gmean2'] = gmean2_traj
    results['valloss'] = valloss_traj
    results['valacc'] = valacc_traj
    results['valkappa'] = valkappa_traj
    results['valauc'] = valauc_traj
    results['valauc2'] = valauc2_traj
    results['valgmean'] = valgmean_traj
    results['valgmean2'] = valgmean2_traj
    results['testloss'] = testloss_traj
    results['testacc'] = testacc_traj
    results['testkappa'] = testkappa_traj
    results['testauc'] = testauc_traj
    results['testauc2'] = testauc2_traj
    results['testgmean'] = testgmean_traj
    results['testgmean2'] = testgmean2_traj
    results['lr_traj'] = lr_traj
    
    return model, results

def load_data(data_type, seed, encoding='isi'):
    if encoding == 'sce':
        file_path = './'+data_type+'data_sce_seed'+str(seed)+'.pkl'
    else:
        file_path = './'+data_type+'data_seed'+str(seed)+'.pkl'
        
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            print("Pickle file loaded successfully!")
        return data
    retina_states = ['randomly_moving_bar',
                     'repeated_natural_movie',
                     'unique_natural_movie',
                     'white_noise_checkerboard',]
    if data_type == 'retina03':
        X_trainval, X_test, y_trainval, y_test, gr_trainval, gr_test = load_retina(random_seed=seed, state1 = retina_states[0], state2 = retina_states[3], encoding=encoding, bin_size=4000)
    elif data_type == 'retina01':
        X_trainval, X_test, y_trainval, y_test, gr_trainval, gr_test = load_retina(random_seed=seed, state1 = retina_states[0], state2 = retina_states[1], encoding=encoding, bin_size=4000)
    elif data_type == 'retina02':
        X_trainval, X_test, y_trainval, y_test, gr_trainval, gr_test = load_retina(random_seed=seed, state1 = retina_states[0], state2 = retina_states[2], encoding=encoding, bin_size=4000)
    elif data_type == 'retina12':
        X_trainval, X_test, y_trainval, y_test, gr_trainval, gr_test = load_retina(random_seed=seed, state1 = retina_states[1], state2 = retina_states[2], encoding=encoding, bin_size=4000)
    elif data_type == 'retina13':
        X_trainval, X_test, y_trainval, y_test, gr_trainval, gr_test = load_retina(random_seed=seed, state1 = retina_states[1], state2 = retina_states[3], encoding=encoding, bin_size=4000)
    elif data_type == 'retina23':
        X_trainval, X_test, y_trainval, y_test, gr_trainval, gr_test = load_retina(random_seed=seed, state1 = retina_states[2], state2 = retina_states[3], encoding=encoding, bin_size=4000)
    elif data_type == 'retinaall':
        X_trainval, X_test, y_trainval, y_test, gr_trainval, gr_test = load_retina_all(random_seed=seed, encoding=encoding, bin_size=4000)
    else:
        raise NotImplementedError

    data = {
        'X_trainval': X_trainval,
        'X_test': X_test,
        'y_trainval': y_trainval,
        'y_test': y_test, 
        'gr_trainval': gr_trainval, 
        'gr_test': gr_test
    }
    # Save the dictionary as a pickle file
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    return data

def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))
    
def log1p_func(X_trainval, X_test, use_log):
    if use_log:
        return signed_log1p(X_trainval), signed_log1p(X_test)
    return X_trainval, X_test

def get_processed_data(data_type, seed, X_trainval, train_idx, y_trainval, X_test, y_test, features, use_log, bins):
    X_train = X_trainval[train_idx]
    y_train = y_trainval[train_idx]

    processed_data_dict = {}
    if 'isi' in features:
        processed_data_dict['isi'] = log1p_func(X_trainval, X_test, use_log)
    if 'isi_sdfa' in features:
        file_path = './features/'+data_type+'_isi_sdfa'+str(bins)+'_seed'+str(seed)+'.pickle'
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                X_trainval_sdfa, X_test_sdfa = pickle.load(f)
        else:
            X_trainval_sdfa = sample_distance_feature_from_agg(X_trainval, bins, X_train.reshape(-1))
            X_test_sdfa = sample_distance_feature_from_agg(X_test, bins, X_train.reshape(-1))
            with open(file_path, 'wb') as f:
                pickle.dump((X_trainval_sdfa, X_test_sdfa), f)
        processed_data_dict['isi_sdfa'] = log1p_func(X_trainval_sdfa, X_test_sdfa, use_log)
    if 'isi_sdfa0c' in features:
        file_path = './features/'+data_type+'_isi_sdfa0c'+str(bins)+'_seed'+str(seed)+'.pickle'
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                X_trainval_sdfa0c, X_test_sdfa0c = pickle.load(f)
        else:
            X_trainval_sdfa0c = sample_distance_feature_from_agg(X_trainval, bins, X_train[y_train==0].reshape(-1))
            X_test_sdfa0c = sample_distance_feature_from_agg(X_test, bins, X_train[y_train==0].reshape(-1))
            with open(file_path, 'wb') as f:
                pickle.dump((X_trainval_sdfa0c, X_test_sdfa0c), f)
        processed_data_dict['isi_sdfa0c'] = log1p_func(X_trainval_sdfa0c, X_test_sdfa0c, use_log)
    if 'isi_sdfa1c' in features:
        file_path = './features/'+data_type+'_isi_sdfa1c'+str(bins)+'_seed'+str(seed)+'.pickle'
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                X_trainval_sdfa1c, X_test_sdfa1c = pickle.load(f)
        else:
            X_trainval_sdfa1c = sample_distance_feature_from_agg(X_trainval, bins, X_train[y_train==1].reshape(-1))
            X_test_sdfa1c = sample_distance_feature_from_agg(X_test, bins, X_train[y_train==1].reshape(-1))
            with open(file_path, 'wb') as f:
                pickle.dump((X_trainval_sdfa1c, X_test_sdfa1c), f)
        processed_data_dict['isi_sdfa1c'] = log1p_func(X_trainval_sdfa1c, X_test_sdfa1c, use_log)
    if 'isi_sdfa2c' in features:
        file_path = './features/'+data_type+'_isi_sdfa2c'+str(bins)+'_seed'+str(seed)+'.pickle'
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                X_trainval_sdfa2c, X_test_sdfa2c = pickle.load(f)
        else:
            X_trainval_sdfa2c = sample_distance_feature_from_agg(X_trainval, bins, X_train[y_train==2].reshape(-1))
            X_test_sdfa2c = sample_distance_feature_from_agg(X_test, bins, X_train[y_train==2].reshape(-1))
            with open(file_path, 'wb') as f:
                pickle.dump((X_trainval_sdfa2c, X_test_sdfa2c), f)
        processed_data_dict['isi_sdfa2c'] = log1p_func(X_trainval_sdfa2c, X_test_sdfa2c, use_log)
    if 'isi_sdfa3c' in features:
        file_path = './features/'+data_type+'_isi_sdfa3c'+str(bins)+'_seed'+str(seed)+'.pickle'
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                X_trainval_sdfa3c, X_test_sdfa3c = pickle.load(f)
        else:
            X_trainval_sdfa3c = sample_distance_feature_from_agg(X_trainval, bins, X_train[y_train==3].reshape(-1))
            X_test_sdfa3c = sample_distance_feature_from_agg(X_test, bins, X_train[y_train==3].reshape(-1))
            with open(file_path, 'wb') as f:
                pickle.dump((X_trainval_sdfa3c, X_test_sdfa3c), f)
        processed_data_dict['isi_sdfa3c'] = log1p_func(X_trainval_sdfa3c, X_test_sdfa3c, use_log)
        

    return processed_data_dict

def gather_features(processed_data_dict, use_CNN):
    if use_CNN:
        trainval_data = [v[0][:,np.newaxis,:] for v in processed_data_dict.values()]
        trainval_data = np.concatenate(trainval_data, axis=1)
        test_data = [v[1][:,np.newaxis,:] for v in processed_data_dict.values()]
        test_data = np.concatenate(test_data, axis=1)
    else:
        trainval_data = [v[0] for v in processed_data_dict.values()]
        trainval_data = np.hstack(trainval_data)
        test_data = [v[1] for v in processed_data_dict.values()]
        test_data = np.hstack(test_data)
    return trainval_data, test_data

def scale_data(data, datatype, scaler_X):
    if len(data.shape) == 3:
        n_samples, rows, cols = data.shape
        if datatype == 'train':
            data_sc = scaler_X.fit_transform(data.reshape(n_samples, -1)).reshape(n_samples, rows, cols)
        else:
            data_sc = scaler_X.transform(data.reshape(n_samples, -1)).reshape(n_samples, rows, cols)
    else:
        if datatype == 'train':
            data_sc = scaler_X.fit_transform(data)
        else:
            data_sc = scaler_X.transform(data)
    return data_sc

def train_model(args):
    data_type, features, model_type, use_val, use_log, use_balancing, seed, device \
    = args.data_type, args.features, args.model_type, \
    args.use_val, args.use_log, args.use_balancing, \
    args.seed, args.dev
    dataseed = args.dataseed
    n_epochs, batch_size, lr, wd = args.n_epochs, args.batch_size, args.lr, args.wd
    use_CNN = False
    use_NN = True
    features2 = args.features2
    use_CNN_for_additional = args.use_CNN_for_additional
    if model_type in ['RF']:
        use_NN = False
    if model_type in ['1dCNN', 'InceptionTimePlus', 'FCNPlus', 'ResNetPlus', 'XceptionTimePlus']:
        use_CNN = True

    # Load data
    data = load_data(data_type, dataseed)
    X_trainval, X_test, y_trainval, y_test, gr_trainval, gr_test = (
        data['X_trainval'],  # Features for training and validation
        data['X_test'],      # Features for testing
        data['y_trainval'],  # Labels for training and validation
        data['y_test'],      # Labels for testing
        data['gr_trainval'], # Group labels for training and validation
        data['gr_test']      # Group labels for testing
    )

    if use_val:
        # Initialize GroupShuffleSplit
        splitter = GroupShuffleSplit(test_size=args.val_ratio, n_splits=1, random_state=args.valsplit_seed)
        # Split data
        for train_idx, val_idx in splitter.split(X_trainval, y_trainval, gr_trainval):
            X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
            y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]
            gr_train, gr_val = gr_trainval[train_idx], gr_trainval[val_idx]
        print(len(train_idx), len(val_idx), len(X_test))
    else:
        train_idx = np.arange(len(X_trainval))
        X_train = X_trainval
        y_train = y_trainval
        X_val = None
        y_val = None
        
    # convert to features
    if args.bins is None:
        bins = X_train.shape[1] # this may be set from arguments later
    else:
        bins = args.bins
    processed_data_dict = get_processed_data(data_type, dataseed, X_trainval, train_idx, y_trainval, X_test, y_test, features, use_log, bins)
    trainval_data, test_data = gather_features(processed_data_dict, use_CNN)
    print([key for key in processed_data_dict.keys()])
    print(trainval_data.shape, test_data.shape)
    if len(features2) > 0:
        processed_data_dict2 = get_processed_data(data_type, dataseed, X_trainval, train_idx, y_trainval, X_test, y_test, features2, use_log, bins)
        trainval_data2, test_data2 = gather_features(processed_data_dict2, use_CNN)
        print([key for key in processed_data_dict2.keys()])
        print(trainval_data2.shape, test_data2.shape)
    
    # train models
    if model_type == 'RF':
        model = RandomForestClassifier(
            n_estimators=500,
            random_state=seed,
            max_depth=10,
            n_jobs=-1)
        train_data = trainval_data[train_idx]
        model.fit(train_data, y_train)
        y_pred = model.predict(test_data)
        acc = (y_pred == y_test).sum()/len(y_pred)
        roc_auc = roc_auc_score(y_test, model.predict_proba(test_data)[:, 1])
        kappa = cohen_kappa_score(y_test, y_pred)
        gmean = geometric_mean_score(y_test.ravel(), y_pred.ravel())
        res = {'testacc': acc, 'testauc': roc_auc, 'testkappa': kappa, 'testgmean': gmean}
        used_data = train_data, test_data
        print(train_data.shape)
    elif use_NN:
        start = time.time()
        # Standardize the data
        scaler_X = StandardScaler()
        train_data = trainval_data[train_idx]
        train_data_sc = scale_data(train_data, 'train', scaler_X)
        test_data_sc = scale_data(test_data, 'test', scaler_X)
        if use_val:
            val_data_sc = scale_data(trainval_data[val_idx], 'val', scaler_X)
        else:
            val_data_sc = None
        if len(features2) == 0:
            print('use train_cls')
            model, res = train_cls(train_data_sc, y_train, test_data_sc, y_test, device, val_data_sc, y_val, 
                                   use_balancing=use_balancing,
                                   model_type=model_type, n_epochs = n_epochs, batch_size = batch_size, lr=lr, wd=wd, 
                                   alg=args.algorithm, schedule=args.scheduler, batch_standardize=args.use_batch_standardize,
                                   pct_start=args.pct_start)
            used_data = train_data_sc, test_data_sc, val_data_sc
        else:
            # Standardize the data
            scaler_X2 = StandardScaler()
            train_data2 = trainval_data2[train_idx]
            train_data2_sc = scale_data(train_data2, 'train', scaler_X2)
            test_data2_sc = scale_data(test_data2, 'test', scaler_X2)
            if use_val:
                val_data2_sc = scale_data(trainval_data2[val_idx], 'val', scaler_X2)
            else:
                val_data2_sc = None
            print('use train_cls2')
            model, res = train_cls2(train_data_sc, train_data2_sc, y_train, 
                                    test_data_sc, test_data2_sc, y_test, device,
                                    val_data_sc, val_data2_sc, y_val, 
                                   use_balancing=use_balancing,
                                   model_type=model_type, n_epochs = n_epochs, batch_size = batch_size, lr=lr, wd=wd, 
                                   alg=args.algorithm, schedule=args.scheduler, batch_standardize=args.use_batch_standardize,
                                    use_CNN_for_additional=use_CNN_for_additional, pct_start=args.pct_start)
            used_data = train_data_sc, train_data2_sc, test_data_sc, test_data2_sc, val_data_sc, val_data2_sc
        print(f"Elapsed time: {time.time() - start} sec.")
    else:
        raise NotImplementedError
    res.update({
        "data_type": data_type,
        "features": features,
        "model_type": model_type,
        "use_val": use_val,
        "use_log": use_log,
        "use_balancing": use_balancing,
        "seed": seed,
        "device": device
    })
    # Add args to res
    res.update(vars(args))

    return model, res, used_data