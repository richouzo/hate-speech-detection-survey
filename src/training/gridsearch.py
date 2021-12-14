import time
import argparse
import datetime

import numpy as np
import pandas as pd

import torch

import itertools
import yaml

from src.utils.preprocess_utils import get_datasets, get_dataloaders
from src.utils.utils import GRIDSEARCH_CSV
from src.training.main import main

def get_gridsearch_config(config_path):
    with open(config_path, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    hyperparameters = config['hyperparameters']
    print('hyperparameters keys', list(hyperparameters.keys()))

    all_config_list = []
    for param_name in hyperparameters.keys():
        all_config_list.append(hyperparameters[param_name])

    return all_config_list

def gridsearch(config_path, training_data, testset_data, test_labels_data, do_save, device):
    all_config_list = get_gridsearch_config(config_path)

    training_remaining = np.prod([len(config) for config in all_config_list])
    print('Training to do:', training_remaining)

    # Save gridsearch training to csv
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    csv_path = GRIDSEARCH_CSV+'results_{}.csv'.format(current_time)
    results_dict = {'model_type': [], 
                    'optimizer_type': [], 
                    'loss_criterion': [], 
                    'lr': [], 
                    'epochs': [], 
                    'batch_size': [], 
                    'patience_es': [], 
                    'scheduler_type': [],
                    'patience_lr': [], 
                    'save_condition': [],
                    'fix_length': [],
                    'best_epoch': [], 
                    'train_loss': [], 
                    'val_loss': [], 
                    'train_acc': [], 
                    'val_acc': [], 
                    'test_acc': [], 
                    'end_time': []}

    # Start gridsearch
    prev_model_type = None
    start_time = time.time()
    for params in itertools.product(*all_config_list):
        # /!\ Has to be in the same order as in the config.yaml file /!\ #
        model_type, optimizer_type, \
        loss_criterion, lr, epochs, \
        batch_size, patience_es, \
        scheduler_type, patience_lr, \
        save_condition, fix_length, context_size, pyramid, fcs, batch_norm, alpha = params

        if prev_model_type != model_type:
            print("prev_model_type", prev_model_type)
            print("model_type", model_type)
            print("Changing tokenizer...")
            ENGLISH, tokenizer, train_data, val_data, test_data = get_datasets(training_data, 
                                                                               testset_data, test_labels_data, 
                                                                               model_type, fix_length)
            prev_model_type = model_type

        print('fix_length:', fix_length)
        print('batch_size:', batch_size)
        
        if model_type == 'PyramidCNN':
            print('context_size:', context_size)
            print('pyramid:', pyramid)
            print('fcs:', fcs)
            print('batch_norm:', batch_norm)
            print('alpha:', alpha)
        dataloaders = get_dataloaders(train_data, val_data, test_data, batch_size, device)

        history_training = main(dataloaders, ENGLISH, model_type, optimizer_type, 
                               loss_criterion, lr, batch_size, epochs, patience_es, 
                               do_save, device, 
                               do_print=False, training_remaining=training_remaining, 
                               scheduler_type=scheduler_type, patience_lr=patience_lr, 
                               save_condition=save_condition, fix_length=fix_length, 
                               context_size=context_size, pyramid=pyramid, fcs=fcs,
                               batch_norm=batch_norm, alpha=alpha)

        # Save training results to csv
        best_epoch = history_training['best_epoch']
        for key in results_dict.keys():
            if key in ['train_loss', 'val_loss', 'train_acc', 'val_acc']:
                results_dict[key].append(history_training[key][best_epoch])
            elif key == 'epochs':
                results_dict[key].append(epochs)
            elif key == 'batch_size':
                results_dict[key].append(batch_size)
            else:
                results_dict[key].append(history_training[key])

        results_csv = pd.DataFrame(data=results_dict)
        results_csv.to_csv(csv_path)

        training_remaining -= 1

    time_elapsed = time.time() - start_time
    print('\nGridsearch complete in {:.0f}m {:.0f}s'.format(
          time_elapsed // 60, time_elapsed % 60))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", help="unprocessed OLID training dataset", default="data/training_data/offenseval-training-v1.tsv")
    parser.add_argument("--testset_data", help="unprocessed OLID testset dataset", default="data/test_data/testset-levela.tsv")
    parser.add_argument("--test_labels_data", help="unprocessed OLID test labels dataset", default="data/test_data/labels-levela.csv")
    parser.add_argument("--config_path", help="gridsearch config", default="gridsearch_config.yml")
    parser.add_argument("--do_save", default=1, help="1 for saving stats and figures, else 0", type=int)
    parser.add_argument("--device", default='' , help="cpu or cuda for gpu")

    args = parser.parse_args()

    # Data processing
    training_data = args.training_data
    testset_data = args.testset_data
    test_labels_data = args.test_labels_data
    config_path = args.config_path

    # Hyperparameters
    do_save = args.do_save

    if args.device in ['cuda', 'cpu']:
        device = args.device
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("Device:", device)

    gridsearch(config_path, training_data, testset_data, test_labels_data, do_save, device)
