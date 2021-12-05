import os
import re
import argparse

import numpy as np
import pandas as pd

from utils import STATS_CSV

def get_highest_lowest_metric_indexes(stats_df, stats_metric='loss', stats_topk=5):
    assert stats_metric in ['prob', 'loss']
    sorted_stats_df = final_stats_df.sort_values(by=[stats_metric])
    lowest_stats_df = sorted_stats_df[:stats_topk]
    highest_stats_df = sorted_stats_df[-stats_topk:][::-1]

    return (lowest_stats_df, highest_stats_df)

def main_test(dataloaders, phase, field, model_type, csv_path, saved_model_path, loss_criterion, device):
    print()
    print('model_type:', model_type)
    print('loss_criterion:', loss_criterion)
    print()

    # Instanciate model 
    model = load_model(model_type, field, device)
    model = load_trained_model(model, saved_model_path, device)

    print("Model {} loaded on {}".format(model_type, device))

    ### Define dictionary for stats results ###
    stats_dict = {'original_index': [], 
                  'text': [], 
                  'true_label': [], 
                  'pred_label': [], 
                  'prob': [], 
                  'loss': []}

    ### Testing ###
    stats_dict = test_model_and_save_stats(model=model, loss_criterion=loss_criterion, dataloaders=dataloaders, 
                                           phase=phase, field=field, stats_dict=stats_dict)

    stats_df = pd.DataFrame(data=stats_dict).reset_index(drop=True)

    stats_df.to_csv(csv_path)

    return stats_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", help="unprocessed OLID training dataset", default="data/training_data/offenseval-training-v1.tsv")
    parser.add_argument("--testset_data", help="unprocessed OLID testset dataset", default="data/test_data/testset-levela.tsv")
    parser.add_argument("--test_labels_data", help="unprocessed OLID test labels dataset", default="data/test_data/labels-levela.csv")
    parser.add_argument("--model", help="model to use. Choices are: BasicLSTM, ...", default='BiLSTM')
    parser.add_argument("--saved_model_path", help="path to trained model", default='saved_models/BiLSTM_2021-12-03_23-58-08_trained_testAcc=0.5561.pth')
    parser.add_argument("--loss_criterion", help="loss function: bceloss, crossentropy", default='bcelosswithlogits')
    parser.add_argument("--device", default='', help="cpu or cuda for gpu")
    parser.add_argument("--stats_metric", default='loss', help="metric to retrieve stats")
    parser.add_argument("--stats_topk", default=5, help="topk indexes to retrieve", type=int)
    parser.add_argument("--stats_label", default=0, help="label indexes to retrieve", type=int)

    args = parser.parse_args()

    # Data processing
    training_data = args.training_data
    testset_data = args.testset_data
    test_labels_data = args.test_labels_data

    # Hyperparameters
    batch_size = 1
    phase = "test"
    saved_model_path = args.saved_model_path
    loss_criterion = args.loss_criterion
    model_type = args.model

    # Get model_id
    regex = '\d+-\d+-\d+_\d+-\d+-\d+'
    find_list = re.findall(regex, saved_model_path)
    assert len(find_list) > 0, "Cannot find model_id (YYYY-MM-DD_HH-MM-SS) in saved_model_path's filename"
    model_id = find_list[0]
    print("model_id:", model_id)

    # Stats parameters
    stats_metric = args.stats_metric
    stats_topk = args.stats_topk
    stats_label = args.stats_label

    csv_path = STATS_CSV+'stats_{}_{}_{}_{}.csv'.format(model_type, model_id, phase, loss_criterion)

    if not os.path.exists(csv_path):
        print("Stats csv does not exist yet, starting the test pipeline...")

        import spacy
        import torch
        from preprocess_utils import get_datasets, get_dataloaders
        from train import test_model_and_save_stats
        from utils import load_model, load_trained_model

        if args.device in ['cuda', 'cpu']:
            device = args.device
        else:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("Device:", device)

        field, train_data, val_data, test_data = get_datasets(training_data, testset_data, test_labels_data)
        dataloaders = get_dataloaders(train_data, val_data, test_data, batch_size, device)

        stats_df = main_test(dataloaders, phase, field, model_type, csv_path, 
                              saved_model_path, loss_criterion, device)

    else:
        print("Stats csv already exists, retrieving csv...")
        stats_df = pd.read_csv(csv_path, index_col=0)


    # Get indexes
    final_stats_df = stats_df[stats_df['true_label'] == stats_label].reset_index(drop=True)

    lowest_stats_df, highest_stats_df = get_highest_lowest_metric_indexes(final_stats_df, 
                                                                          stats_metric=stats_metric, 
                                                                          stats_topk=stats_topk)
    print('lowest_stats_df')
    print(lowest_stats_df)
    print('highest_stats_df')
    print(highest_stats_df)
