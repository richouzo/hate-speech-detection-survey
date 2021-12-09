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
         batch_size, epochs, patience_es, do_save, device, do_print=False, 
         scheduler_type='', patience_lr=5,  
         training_remaining=1, save_condition='acc', fix_length=None,
         context_size=2, pyramid=[64,128,256], fcs=[64,128], batch_norm=1, alpha=0.2):

    print('model_type:', model_type)
    print('optimizer_type:', optimizer_type)
    print('loss_criterion:', loss_criterion)
    print('learning rate:', lr)
    print('epochs:', epochs)
    print('patience_es:', patience_es)
    print('scheduler_type:', scheduler_type)
    print('patience_lr:', patience_lr)
    print('save_condition:', save_condition)
    print()


    # Instantiate model 
    model = load_model(model_type, field, device, fix_length=fix_length,
            context_size=context_size, pyramid=pyramid, fcs=fcs,
            batch_norm=batch_norm, alpha=alpha)


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

    if optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else: # Default to Adam
        optimizer = optim.Adam(model.parameters(), lr=lr)

    print('Optimizer used: {}'.format(optimizer))

    if scheduler_type == 'reduce_lr_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience_lr)
    elif scheduler_type == 'linear_schedule_with_warmup':
        import transformers
        train_length = len(dataloaders['train'].dataset)
        num_training_steps = round((train_length/batch_size)*epochs)
        num_warmup_steps = round(0.1*num_training_steps)
        print('num_training_steps', num_training_steps)
        print('num_warmup_steps', num_warmup_steps)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        scheduler = None

    print('Scheduler used: {}'.format(scheduler))

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
                        'patience_es': patience_es,
                        'scheduler_type': scheduler_type,
                        'patience_lr': patience_lr,
                        'save_condition': save_condition,
                        'fix_length': fix_length}


    ### Training phase ###
    model, history_training = train_model(model, criterion, optimizer, 
                                          dataloaders, history_training, 
                                          num_epochs=epochs, patience_es=patience_es, 
                                          scheduler=scheduler, 
                                          training_remaining=training_remaining, 
                                          save_condition=save_condition)


    ### Testing ###
    history_training = test_model(model=model, history_training=history_training, 
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
    parser.add_argument("--patience_es", default=5, help="nb epochs before early stopping", type=int)
    parser.add_argument("--patience_lr", default=5, help="nb epochs before lr scheduler", type=int)
    parser.add_argument("--scheduler_type", default='', help="reduce_lr_on_plateau, linear_schedule_with_warmup")
    parser.add_argument("--do_save", default=1, help="1 for saving stats and figures, else 0", type=int)
    parser.add_argument("--save_condition", help="save model with"+\
                        " condition on best val_acc (acc) or lowest val_loss(loss)", default='acc')
    parser.add_argument("--device", default='' , help="cpu or cuda for gpu")
    parser.add_argument("--fix_length", default=None, type=int, help="fix length of max number of words per sentence, take max if None")
    parser.add_argument("--context_size", default=2, type=int, help="")
    parser.add_argument('--pyramid', default="256", help='delimited list for pyramid input', type=str)
    parser.add_argument('--fcs', default="128,256", help='delimited list for fcs input', type=str)
    parser.add_argument("--batch_norm", default=1, type=int, help="")
    parser.add_argument("--alpha", default=0.8, type=int, help="")

    args = parser.parse_args()

    # Data processing
    training_data = args.training_data
    testset_data = args.testset_data
    test_labels_data = args.test_labels_data

    # Hyperparameters
    model_type = args.model
    optimizer_type = args.optimizer_type
    loss_criterion = args.loss_criterion
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    patience_es = args.patience_es

    # Scheduler
    scheduler_type = args.scheduler_type
    patience_lr = args.patience_lr

    # Saving condition by acc or loss
    save_condition = args.save_condition
    do_save = args.do_save

    # HybridLSTMCNN
    fix_length = args.fix_length

    # PyramidCNN parameters
    context_size = args.context_size
    pyramid = [int(item) for item in args.pyramid.split(',')]
    fcs = [int(item) for item in args.fcs.split(',')]
    batch_norm = args.batch_norm
    alpha = args.alpha

    if args.device in ['cuda', 'cpu']:
        device = args.device
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("Device:", device)

    field, tokenizer, train_data, val_data, test_data = get_datasets(training_data, testset_data, test_labels_data, model_type, fix_length)

    dataloaders = get_dataloaders(train_data, val_data, test_data, batch_size, device)

    main(dataloaders, field, model_type, optimizer_type, loss_criterion, lr, 
         batch_size, epochs, patience_es, do_save, device, do_print=True, save_condition=save_condition, 
         scheduler_type=scheduler_type, patience_lr=patience_lr, fix_length=fix_length, 
         context_size=context_size, pyramid=pyramid, fcs=fcs,
         batch_norm=batch_norm, alpha=alpha)
