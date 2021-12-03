import tqdm
import argparse
import numpy as np

import spacy
import pandas as pd
from sklearn.metrics import f1_score

import torch
from torchtext.legacy.data import Field, LabelField, TabularDataset, BucketIterator
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset

from torch import optim
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from preprocess_utils import *
from train import train_model, test_model
from models import BasicLSTM

from utils import save_model, plot_training, plot_cm, classif_report


spacy_en = spacy.load("en_core_web_sm")

def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def get_dataloaders(training_data, testset_data, test_labels_data, batch_size, device):
    # preprocessing of the train/validation tweets, then test tweets
    tweets, classes = format_training_file(training_data)
    tweets_test, y_test = format_test_file(testset_data, test_labels_data)
    print("file loaded and formatted..")
    train_val_split_tocsv(tweets, classes, val_size=0.2)
    test_tocsv(tweets_test, y_test)
    print("data split into train/val/test")

    ENGLISH, LABEL, train_data, val_data, test_data = create_fields_dataset(tokenizer)

    # build vocabularies using training set
    print("fields and dataset object created")
    ENGLISH.build_vocab(train_data, max_size=10000, min_freq=2)
    LABEL.build_vocab(train_data)

    print("vocabulary built..")
    train_iterator, val_iterator = create_iterators(train_data, val_data, batch_size, device, shuffle=True)
    _, test_iterator = create_iterators(train_data, test_data, batch_size, device, shuffle=False)
    print("dataloaders created..")

    dataloaders = {}
    dataloaders['train'] = train_iterator
    dataloaders['val'] = val_iterator
    dataloaders['test'] = test_iterator

    return (ENGLISH, dataloaders)

def main(dataloaders, ENGLISH, model_type, optimizer_type, loss_criterion, lr, 
         epochs, patience_es, do_save, device, do_print=False):
    #instanciate model (all models need to be added here)
    if model_type == 'MORE MODELS NAMES':
        model = None #other models here
    elif model_type == 'BasicLSTM':
        model = BasicLSTM.BasicLSTM(dim_emb=300, num_words=ENGLISH.vocab.__len__(), 
                                    hidden_dim=128, num_layers=2, output_dim=1)
    else:
        model = None

    model.to(device)

    print("Model {} loaded on {}".format(model_type, device))

    if loss_criterion == 'bceloss':
        criterion = nn.BCELoss()
    elif loss_criterion == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    else: # Default to BCELoss
        criterion = nn.BCELoss()

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
                        'val_acc': []}


    ### Training phase ###
    model, history_training = train_model(model, criterion, optimizer, 
                                          dataloaders, history_training, 
                                          num_epochs=epochs, patience_es=patience_es)


    ### Testing ###
    history_training = test_model(model=model, history_training=history_training, criterion=criterion, 
                                  dataloaders=dataloaders)


    ### Save the model ###
    save_model(model=model, hist=history_training, 
               trained_models_path=SAVED_MODELS_PATH, model_type=model_type, 
               do_save=do_save, do_print=do_print)


    ### Plot the losses ###
    plot_training(hist=history_training, figures_path=FIGURES_PATH, 
                  model_type=model_type, do_save=do_save, do_print=do_print)


    ### Plotting the CM ###
    plot_cm(hist=history_training, figures_path=FIGURES_PATH, 
            model_type=model_type, do_save=do_save, do_print=do_print)


    ### Give the classification report ###
    if do_print: classif_report(hist=history_training)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", help="unprocessed OLID training dataset", default="data/training_data/offenseval-training-v1.tsv")
    parser.add_argument("--testset_data", help="unprocessed OLID testset dataset", default="data/test_data/testset-levela.tsv")
    parser.add_argument("--test_labels_data", help="unprocessed OLID test labels dataset", default="data/test_data/labels-levela.csv")
    parser.add_argument("--model", help="model to use. Choices are: BasicLSTM, ...", default='BasicLSTM')
    parser.add_argument("--batch_size", help="batch size", type=int, default=128)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-3)
    parser.add_argument("--optimizer_type", help="optimizer: adam, sgd", default='adam')
    parser.add_argument("--loss_criterion", help="loss function: bceloss, crossentropy", default='bceloss')
    parser.add_argument("--epochs", default=10, help="cpu or cuda for gpu", type=int)
    parser.add_argument("--patience_es", default=2, help="nb epochs before early stopping", type=int)
    parser.add_argument("--do_save", default=1, help="1 for saving stats and figures, else 0", type=int)
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

    if args.device in ['cuda', 'cpu']:
        device = args.device
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("Device:", device)

    ENGLISH, dataloaders = get_dataloaders(training_data, testset_data, test_labels_data, batch_size, device)

    main(dataloaders, ENGLISH, model_type, optimizer_type, loss_criterion, lr, 
         epochs, patience_es, do_save, device, do_print=True)
