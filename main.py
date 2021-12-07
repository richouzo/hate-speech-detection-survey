import tqdm
import argparse
import datetime

import spacy
import pandas as pd
from sklearn.metrics import f1_score

import torch

from torch import optim
import torch.nn as nn

from preprocess_utils import get_datasets, get_dataloaders
from train import train_model, test_model

from utils import load_model, save_model, plot_training, plot_cm, classif_report

def main(dataloaders, field, model_type, optimizer_type, loss_criterion, lr, 
         epochs, patience_es, do_save, device, do_print=False, training_remaining=1, save_condition='acc'):
    print()
    print('model_type:', model_type)
    print('optimizer_type:', optimizer_type)
    print('loss_criterion:', loss_criterion)
    print('learning rate:', lr)
    print('epochs:', epochs)
    print('patience_es:', patience_es)
    print()

    # Instantiate model 
    model = load_model(model_type, field, device)

    print("Model {} loaded on {}".format(model_type, device))

    if loss_criterion == 'bceloss':
        criterion = nn.BCELoss()
    elif loss_criterion == 'bcelosswithlogits':
        criterion = nn.BCEWithLogitsLoss()
    elif loss_criterion == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    else: # Default to BCEWithLogitsLoss
        criterion = nn.BCEWithLogitsLoss()

    print('Loss used: {}'.format(criterion))

    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else: # Default to Adam
        optimizer = optim.Adam(model.parameters(), lr=lr)

    print('Optimizer used: {}'.format(optimizer))


    ### Define dictionary for training info ###
    history_training = {'train_loss': [],
                        'val_loss': [],
                        'train_acc': [],
                        'val_acc': [], 
                        'model_type': model_type,
                        'optimizer_type': optimizer_type,
                        'loss_criterion': loss_criterion,
                        'lr': lr,
                        'epochs': epochs,
                        'patience_es': patience_es}


    ### Training phase ###
    model, history_training = train_model(model, criterion, optimizer, 
                                          dataloaders, history_training, 
                                          num_epochs=epochs, patience_es=patience_es, 
                                          training_remaining=training_remaining, 
                                          save_condition=save_condition)


    ### Testing ###
    history_training = test_model(model=model, history_training=history_training, criterion=criterion, 
                                  dataloaders=dataloaders)

    end_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    history_training['end_time'] = end_time

    ### Save the model ###
    save_model(model=model, hist=history_training,
               model_type=model_type, 
               do_save=do_save, do_print=do_print)


    ### Plot the losses ###
    plot_training(hist=history_training, 
                  model_type=model_type, 
                  do_save=do_save, do_print=do_print)


    ### Plotting the CM ###
    plot_cm(hist=history_training, 
            model_type=model_type, 
            do_save=do_save, do_print=do_print)


    ### Give the classification report ###
    if do_print: classif_report(hist=history_training)

    return history_training


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", help="unprocessed OLID training dataset", default="data/training_data/offenseval-training-v1.tsv")
    parser.add_argument("--testset_data", help="unprocessed OLID testset dataset", default="data/test_data/testset-levela.tsv")
    parser.add_argument("--test_labels_data", help="unprocessed OLID test labels dataset", default="data/test_data/labels-levela.csv")
    parser.add_argument("--model", help="model to use. Choices are: BasicLSTM, ...", default='BasicLSTM')
    parser.add_argument("--batch_size", help="batch size", type=int, default=128)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-3)
    parser.add_argument("--optimizer_type", help="optimizer: adam, sgd", default='adam')
    parser.add_argument("--loss_criterion", help="loss function: bceloss, crossentropy", default='bcelosswithlogits')
    parser.add_argument("--epochs", default=10, help="cpu or cuda for gpu", type=int)
    parser.add_argument("--patience_es", default=2, help="nb epochs before early stopping", type=int)
    parser.add_argument("--do_save", default=1, help="1 for saving stats and figures, else 0", type=int)
    parser.add_argument("--save_condition", help="save model with"+\
                        " condition on best val_acc (acc) or lowest val_loss(loss)", default='acc')
    parser.add_argument("--device", default='' , help="cpu or cuda for gpu")

    args = parser.parse_args()

    # Data processing
    training_data = args.training_data
    testset_data = args.testset_data
    test_labels_data = args.test_labels_data

    # Hyperparameters
    batch_size = args.batch_size
    epochs = args.epochs
    patience_es = args.patience_es
    lr = args.lr
    optimizer_type = args.optimizer_type
    loss_criterion = args.loss_criterion
    model_type = args.model
    do_save = args.do_save
    save_condition = args.save_condition

    if args.device in ['cuda', 'cpu']:
        device = args.device
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("Device:", device)

    field, train_data, val_data, test_data = get_datasets(training_data, testset_data, test_labels_data)

    dataloaders = get_dataloaders(train_data, val_data, test_data, batch_size, device)

    main(dataloaders, field, model_type, optimizer_type, loss_criterion, lr, 
         epochs, patience_es, do_save, device, do_print=True, save_condition=save_condition)
