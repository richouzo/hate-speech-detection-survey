import datetime
import numpy as np

import torch

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
 
from models import BasicLSTM, BiLSTM, Transformers, Hybrid_CNN_LSTM, Hybrid_LSTM_CNN, AutoTransformer, PyramidCNN

SAVED_MODELS_PATH = "saved_models/"
FIGURES_PATH = "figures/"
GRIDSEARCH_CSV = "gridsearch_results/"
STATS_CSV = "stats_results/"

def load_model(model_type, field, device, 
               context_size=0, pyramid=[256, 256], 
               fcs=[128], batch_norm=0, alpha=0.2,
               pad_len=30, pooling_size=2,
               fix_length=None, glove=None):
    """
    Load and return model.
    """
    if model_type == 'BasicLSTM':
        model = BasicLSTM.BasicLSTM(dim_emb=300, num_words=field.vocab.__len__(), 
                                    hidden_dim=128, num_layers=2, output_dim=1)

    elif model_type == 'BiLSTM':
        model = BiLSTM.BiLSTM(dim_emb=300, num_words=field.vocab.__len__(), 
                                    hidden_dim=128, num_layers=2, output_dim=1)
    elif model_type == 'Transformers':
        model = Transformers.Transformers(dim_emb=128, num_words=field.vocab.__len__(), 
                                          hidden_dim=128, num_layers=2, output_dim=1)
    elif model_type == 'AutoTransformer':
        model = AutoTransformer.AutoTransformer(dim_emb=128, num_words=field.vocab.__len__(), 
                                          hidden_dim=128, num_layers=2, output_dim=1, hidden_dropout_prob = 0.5)
    elif model_type == 'DistillBert':
        model = Transformers.DistillBert()

    elif model_type == 'DistillBertEmotion':
        model = Transformers.DistillBertEmotion()

    elif model_type == 'PyramidCNN':
        model = PyramidCNN.PyramidCNN(num_words=field.vocab.__len__(),context_size=context_size, 
                                      pyramid=pyramid, fcs=fcs, batch_norm=batch_norm, alpha=alpha, 
                                      pad_len=pad_len, pooling_size=pooling_size, glove=glove)

    elif model_type == 'HybridCNNLSTM':
	      model = Hybrid_CNN_LSTM.HybridCNNLSTM()
        
    elif model_type == 'HybridLSTMCNN':
	      model = Hybrid_LSTM_CNN.HybridLSTMCNN(fix_length=fix_length)
        

    else:
        model = None
    model.to(device)

    return model

def load_trained_model(model, saved_model_path, device):
    """
    Load and return trained model. Initialize the model first with load_model().
    """
    model.load_state_dict(torch.load(saved_model_path, map_location=device))
    print(f"{saved_model_path} loaded.")
    model.to(device)

    return model

def save_model(model, hist, model_type, do_save, do_print=False):
    """
    Save the trained model.
    """
    if do_save:
        end_time = hist['end_time']
        saved_model_path = f"{SAVED_MODELS_PATH}{model_type}_{end_time}_trained_testAcc={hist['test_acc']}.pth"
        torch.save(model.state_dict(), saved_model_path)
        if do_print: print(f"Model saved at {saved_model_path}")

def plot_training(hist, model_type, do_save, do_plot=False, do_print=False):
    """
    Plot the training and validation loss/accuracy.
    """
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title(f'{model_type} - loss')
    ax[0].plot(hist["epochs"], hist["train_loss"], label="Train loss")
    ax[0].plot(hist["epochs"], hist["val_loss"], label="Validation loss")
    ax[1].set_title(f'{model_type} - accuracy')
    ax[1].plot(hist["epochs"], hist["train_acc"], label="Train accuracy")
    ax[1].plot(hist["epochs"], hist["val_acc"], label="Validation accuracy")
    ax[0].legend()
    ax[1].legend()
    if do_save:
        end_time = hist['end_time']
        save_graph_path = f"{FIGURES_PATH}{model_type}_losses&acc_{end_time}_testAcc={hist['test_acc']}.png"
        plt.savefig(save_graph_path)
        if do_print: print(f"Training graph saved at {save_graph_path}")
    if do_plot: plt.show()

def classif_report(hist, list_names=[]):
    """
    Give the classification report from sklearn.
    """
    y_pred = [y for y in hist['y_pred']]
    y_true = [y for y in hist['y_true']]

    nb_classes = len(set(y_true))

    accuracy = round(accuracy_score(y_true, y_pred)*100, 3)
    macro_f1score = round(f1_score(y_true, y_pred, average='macro')*100, 3)
    binary_f1score = round(f1_score(y_true, y_pred, average='binary')*100, 3)
    mse = round(mean_squared_error(y_true, y_pred), 3)
    print(f'Accuracy: {accuracy}%')
    print(f'Macro F1-score: {macro_f1score}%')
    print(f'Binary F1-score: {binary_f1score}%')
    print(f'MSE: {mse}')
    target_names = list_names if list_names else [f'class {i}' for i in range(nb_classes)]
    print(classification_report(y_true, y_pred, target_names=target_names))

def plot_cm(hist, model_type, do_save, do_plot=False, do_print=False):
    """
    Plot the confusion matrix after testing.
    """
    y_pred = [y for y in hist['y_pred']]
    y_true = [y for y in hist['y_true']]

    nb_classes = len(set(y_true))
    end_time = hist['end_time']
    cm_path = f"{FIGURES_PATH}{model_type}_CM_{end_time}_testAcc={hist['test_acc']}.png"

    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index = [i for i in range(nb_classes)], 
                         columns = [i for i in range(nb_classes)])
    plt.figure(figsize = (10,7))
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    sns.heatmap(df_cm, cmap=cmap, annot=True, fmt='.0f')
    plt.title(f"Confusion Matrix for {model_type}")

    if do_save:
        plt.savefig(cm_path)
        if do_print: print(f"Confusion Matrix saved at {cm_path}")
    if do_plot: plt.show()

# From https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, save_condition='loss', path=SAVED_MODELS_PATH+'checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.score_min = np.Inf
        self.delta = delta
        self.path = path
        self.save_condition = save_condition
        assert self.save_condition in ['acc', 'loss']

    def __call__(self, val_loss, val_acc, model):
        if self.save_condition == 'loss':
            score = -val_loss
        elif self.save_condition == 'acc':
            score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_acc, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            if self.save_condition == 'loss':
                print(f'Validation loss decreased ({self.score_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            elif self.save_condition == 'acc':
                print(f'Validation acc increased ({self.score_min:.6f} --> {val_acc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.score_min = val_loss
