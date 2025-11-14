import os
import datetime
import pickle
import argparse
from os.path import join, splitext
import setproctitle
import torch
from spikebench.helpers import set_random_seed
from train_retina import train_model

def train(args):
    model, res, _ = train_model(args)
    string = ''
    for s in args.features:
        string += s+'+' 
    augstring = ''
    modelaugstring = ''
    if len(args.features2) > 0:
        if args.use_CNN_for_additional:
            modelaugstring = '+CNN'
        else:
            modelaugstring = '+MLP'
        augstring = '_aug_'
        for s in args.features2:
            augstring += s+'+' 
    valstr = 'noval'
    logstr = 'nolog'
    balstr = 'nobal'
    if args.use_val:
        valstr = 'val'
    if args.use_log:
        logstr = 'log'
    if args.use_balancing:
        balstr = 'bal'
    string = string[:-1] + augstring[:-1] +'_' + args.model_type+modelaugstring+'_'+valstr+'_'+logstr+'_'+balstr+args.folder_suffix\
    +'_dataseed'+str(args.dataseed)+'_seed'+str(args.seed)+'_valsplitseed'+str(args.valsplit_seed)
    folder = join('./results', args.data_type, string)
    print(folder)
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = join(folder, 'results.pickle')
    filename2 = join(folder, 'model_weights.pth')
    filename3 = join(folder, 'model.pickle')

    with open(filename, 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if args.model_type == 'RF':
        with open(filename3, 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        torch.save(model.state_dict(), filename2)
    
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training ')
     
    parser.add_argument(
        "--data-type",
        type=str,   # Specify that the inputs are strings
        default='retina03',
        help="Data type (retinaall/retina01/retina02/retina03/retina12/retina13/retina23)."
    )
    parser.add_argument(
        "--features",
        nargs="+",  # Accept one or more values
        type=str,   # Specify that the inputs are strings
        default=[],
        required=True,  # Make this argument mandatory
        help="List of features."
    )
    parser.add_argument(
        "--features2",
        nargs="*",  # Accept zero or more values
        type=str,   # Specify that the inputs are strings
        default=[],
        help="List of features to augment."
    )
    parser.add_argument(
        "--model-type",
        type=str,   # Specify that the inputs are strings
        default='1dCNN',
        help="Model type."
    )
    parser.add_argument(
        "--folder-suffix",
        type=str,   # Specify that the inputs are strings
        default='',
        help="suffix for foldername."
    )
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='choose gpu number.')
    parser.add_argument('--dataseed',
                        type=int,
                        default=0,
                        help='random seed for data (default: 0).')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='random seed (default: 0).')
    parser.add_argument("--seeds", nargs='*', type=int, default=[])
    parser.add_argument("--use-val", action='store_true', help="use validation data.")
    parser.add_argument("--use-log", action='store_true', help="use log of features.")
    parser.add_argument("--use-balancing", action='store_true', help="use class balancing.")
    parser.add_argument("--use-batch-standardize", action='store_true', help="use batch standardization.")
    parser.add_argument("--use-CNN-for-additional", action='store_true', help="use CNN for additional features.")
    parser.add_argument('--n_epochs',
                        type=int,
                        default=10,
                        help='the number of epochs.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='batch size.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='learning rate.')
    parser.add_argument('--wd',
                        type=float,
                        default=1e-4,
                        help='weight decay.')
    parser.add_argument('--algorithm',
                        type=str,
                        default='Adam',
                        help='optimization algorithm.')
    parser.add_argument('--scheduler',
                        type=str,
                        default=None,
                        help='lr scheduler.')
    parser.add_argument('--pct-start',
                        type=float,
                        default=0.75,
                        help='the ratio for the flat lr part in flatcosine scheduler.')
    parser.add_argument('--val-ratio',
                        type=float,
                        default=0.2,
                        help='the ratio for validation set.')
    parser.add_argument('--valsplit-seed',
                        type=int,
                        default=0,
                        help='random seed for validation split (default: 0).')
    parser.add_argument('--bins',
                        type=int,
                        default=None,
                        help='the number of bins.')
    
    
    args = parser.parse_args()
    
    # time
    time_now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    setproctitle.setproctitle(time_now)
    print(time_now)
    
    for arg in vars(args):
        print(arg, getattr(args, arg))
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    
    if args.gpu == -1:
        args.dev = 'cpu'
    else:
        args.dev = 'cuda:' + str(args.gpu)
    
    # set random seed to reproduce the work
    if len(args.seeds) == 0:
        print(f"use seed {args.seed}.")
        set_random_seed(args.seed)
        train(args)
    else:
        print(args.seeds)
        for seed in args.seeds:
            args.seed = seed
            set_random_seed(args.seed)
            train(args)