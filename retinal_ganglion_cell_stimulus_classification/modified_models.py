import torch
import torch.nn as nn
from tsai.models.InceptionTimePlus import InceptionTimePlus
from tsai.models.FCNPlus import FCNPlus
from tsai.models.ResNetPlus import ResNetPlus
from tsai.models.XceptionTimePlus import XceptionTimePlus

class ModifiedInceptionTimePlus(nn.Module):
    def __init__(self, c_in, c_out, additional_c_in, additional_seq_len, use_inception_for_additional=False):
        super(ModifiedInceptionTimePlus, self).__init__()
        
        # InceptionTimePlus for main input
        self.main_inception = InceptionTimePlus(c_in=c_in, c_out=128)  # Output: 128 features
        
        # Additional feature processor: InceptionTimePlus or MLP
        self.use_inception_for_additional = use_inception_for_additional
        if use_inception_for_additional:
            self.additional_processor = InceptionTimePlus(c_in=additional_c_in, c_out=64)  # Output: 64 features
        else:
            self.additional_processor = nn.Sequential(
                nn.Linear(additional_c_in * additional_seq_len, 128),  # Flatten input first
                nn.ReLU(),
                nn.Linear(128, 64)  # Final output: 64 features
            )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 256),  # Combine features from both processors
            nn.ReLU(),
            nn.Linear(256, c_out)
        )
    
    def forward(self, x, additional_feature):
        # Process main input
        x = self.main_inception(x)  # Shape: (batch_size, 128)
        
        # Process additional feature
        if self.use_inception_for_additional:
            additional_feature = self.additional_processor(additional_feature)  # Shape: (batch_size, 64)
        else:
            # Flatten additional feature for MLP processing
            additional_feature = additional_feature.view(additional_feature.size(0), -1)
            additional_feature = self.additional_processor(additional_feature)  # Shape: (batch_size, 64)
        
        # Concatenate features
        combined = torch.cat((x, additional_feature), dim=1)  # Shape: (batch_size, 192)
        
        # Pass through the final classifier
        output = self.classifier(combined)
        
        return output

class ModifiedFCNPlus(nn.Module):
    def __init__(self, c_in, c_out, additional_c_in, additional_seq_len, use_fcn_for_additional=False):
        super(ModifiedFCNPlus, self).__init__()
        
        # FCNPlus for main input
        self.main_fcn = FCNPlus(c_in=c_in, c_out=128)  # Output: 128 features
        
        # Additional feature processor: FCNPlus or MLP
        self.use_fcn_for_additional = use_fcn_for_additional
        if use_fcn_for_additional:
            self.additional_processor = FCNPlus(c_in=additional_c_in, c_out=64)  # Output: 64 features
        else:
            self.additional_processor = nn.Sequential(
                nn.Linear(additional_c_in * additional_seq_len, 128),  # Flatten input first
                nn.ReLU(),
                nn.Linear(128, 64)  # Final output: 64 features
            )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 256),  # Combine features from both processors
            nn.ReLU(),
            nn.Linear(256, c_out)
        )
    
    def forward(self, x, additional_feature):
        # Process main input
        x = self.main_fcn(x)  # Shape: (batch_size, 128)
        
        # Process additional feature
        if self.use_fcn_for_additional:
            additional_feature = self.additional_processor(additional_feature)  # Shape: (batch_size, 64)
        else:
            additional_feature = additional_feature.view(additional_feature.size(0), -1)
            additional_feature = self.additional_processor(additional_feature)  # Shape: (batch_size, 64)
        
        # Concatenate features
        combined = torch.cat((x, additional_feature), dim=1)  # Shape: (batch_size, 192)
        
        # Pass through the final classifier
        output = self.classifier(combined)
        
        return output


class ModifiedResNetPlus(nn.Module):
    def __init__(self, c_in, c_out, additional_c_in, additional_seq_len, use_resnet_for_additional=False):
        super(ModifiedResNetPlus, self).__init__()
        
        # ResNetPlus for main input
        self.main_resnet = ResNetPlus(c_in=c_in, c_out=128)  # Output: 128 features
        
        # Additional feature processor: ResNetPlus or MLP
        self.use_resnet_for_additional = use_resnet_for_additional
        if use_resnet_for_additional:
            self.additional_processor = ResNetPlus(c_in=additional_c_in, c_out=64)  # Output: 64 features
        else:
            self.additional_processor = nn.Sequential(
                nn.Linear(additional_c_in * additional_seq_len, 128),  # Flatten input first
                nn.ReLU(),
                nn.Linear(128, 64)  # Final output: 64 features
            )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 256),  # Combine features from both processors
            nn.ReLU(),
            nn.Linear(256, c_out)
        )
    
    def forward(self, x, additional_feature):
        # Process main input
        x = self.main_resnet(x)  # Shape: (batch_size, 128)
        
        # Process additional feature
        if self.use_resnet_for_additional:
            additional_feature = self.additional_processor(additional_feature)  # Shape: (batch_size, 64)
        else:
            additional_feature = additional_feature.view(additional_feature.size(0), -1)
            additional_feature = self.additional_processor(additional_feature)  # Shape: (batch_size, 64)
        
        # Concatenate features
        combined = torch.cat((x, additional_feature), dim=1)  # Shape: (batch_size, 192)
        
        # Pass through the final classifier
        output = self.classifier(combined)
        
        return output

class ModifiedXceptionTimePlus(nn.Module):
    def __init__(self, c_in, c_out, additional_c_in, additional_seq_len, use_xception_for_additional=False):
        super(ModifiedXceptionTimePlus, self).__init__()
        
        # XceptionTimePlus for main input
        self.main_xception = XceptionTimePlus(c_in=c_in, c_out=128)  # Output: 128 features
        
        # Additional feature processor: XceptionTimePlus or MLP
        self.use_xception_for_additional = use_xception_for_additional
        if use_xception_for_additional:
            self.additional_processor = XceptionTimePlus(c_in=additional_c_in, c_out=64)  # Output: 64 features
        else:
            self.additional_processor = nn.Sequential(
                nn.Linear(additional_c_in * additional_seq_len, 128),  # Flatten input first
                nn.ReLU(),
                nn.Linear(128, 64)  # Final output: 64 features
            )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 256),  # Combine features from both processors
            nn.ReLU(),
            nn.Linear(256, c_out)
        )
    
    def forward(self, x, additional_feature):
        # Process main input
        x = self.main_xception(x)  # Shape: (batch_size, 128)
        
        # Process additional feature
        if self.use_xception_for_additional:
            additional_feature = self.additional_processor(additional_feature)  # Shape: (batch_size, 64)
        else:
            additional_feature = additional_feature.view(additional_feature.size(0), -1)
            additional_feature = self.additional_processor(additional_feature)  # Shape: (batch_size, 64)
        
        # Concatenate features
        combined = torch.cat((x, additional_feature), dim=1)  # Shape: (batch_size, 192)
        
        # Pass through the final classifier
        output = self.classifier(combined)
        
        return output