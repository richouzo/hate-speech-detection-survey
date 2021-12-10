import tqdm
import argparse
import numpy as np
import datetime
import time

import spacy
import pandas as pd
from sklearn.metrics import f1_score

from torch import optim
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from preprocess_utils import *
from train import train_model, test_model
from models import BasicLSTM, BiLSTM

from utils import *

import captum
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization

spacy_en = spacy.load("en_core_web_sm")


def batch_model_explainability(model, vocab_stoi, vocab_itos, dataloaders, field, device):
    """
    Using LIME to get qualitative results on words' importance to make
    the decision
    """
    print("\n\n**MODEL EXPLAINABILITY**\n")

    PAD_IND = field.vocab.stoi[field.pad_token] +1 #vocab_stoi[field.pad_token]
    print('PAD_IND', PAD_IND)
    token_reference = TokenReferenceBase(reference_token_idx=PAD_IND)

    lig = LayerIntegratedGradients(model, model.emb)

    # accumalate couple samples in this array for visualization purposes
    vis_data_records_ig = []

    phase = "test"
    model.train()

    nb_batches = len(dataloaders[phase])
    length_phase = len(dataloaders[phase].dataset)

    pbar = tqdm.tqdm([i for i in range(nb_batches)])

    # Iterate over data.
    # batch_size is set to 1.
    for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
        pbar.update()
        pbar.set_description("Processing batch %s" % str(batch_idx+1))
        labels = int(labels)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(True):
            #output = model.forward(inputs)
            #print(output)
            interpret_sentence(model, field, inputs, vocab_stoi, vocab_itos, 
                               device, vis_data_records_ig, token_reference, lig, min_len = 7, label = labels)
        # break

    pbar.close()

    return vis_data_records_ig

def convert_token_to_str(input_token, vocab_stoi, vocab_itos):
    str_input = ""
    for i in range(len(input_token)):
        str_input+=vocab_itos[input_token[i]]+" "
    return str_input

def interpret_sentence(model, field, inputs, vocab_stoi, vocab_itos, device, vis_data_records_ig, token_reference, lig, min_len = 7, label = 0):
    # PAD_IND = vocab_stoi[field.pad_token]
    indexed = [int(inputs[i,0]) for i in range(inputs.shape[0])]
    if len(indexed) < min_len :
        indexed +=[vocab_stoi[field.pad_token]] * (min_len - len(indexed))
    print("indexed", indexed)
    sentence = convert_token_to_str(indexed, vocab_stoi, vocab_itos)
    # print("sentence", sentence)
    text = [vocab_itos[tok] for tok in indexed]
    if len(text) < min_len:
        text += [vocab_itos[field.pad_token]] * (min_len - len(text))
    print("text", text)
    indexed = [vocab_stoi[t] for t in text]
    input_indices = torch.tensor(indexed, device=device)
    model.zero_grad()

    # input_indices = torch.tensor(inputs, device=device)
    input_indices = input_indices.unsqueeze(0)

    # input_indices dim: [sequence_length]
    seq_length = inputs.shape[0]

    input_indices = inputs

    # predict
    # print("inputs indices", input_indices.shape)
    out = model.forward(inputs)
    out = torch.sigmoid(out)
    pred = out.item()
    pred_ind = round(pred)

    # generate reference indices for each sample
    reference_indices = token_reference.generate_reference(seq_length, device=device).unsqueeze(0).permute(1, 0)
    print("ref_indices", reference_indices.shape)

    # compute attributions and approximation delta using layer integrated gradients
    attributions_ig, delta = lig.attribute(input_indices, reference_indices, \
                                           n_steps=500, return_convergence_delta=True)

    class_names = ["Neutral","Hate"]

    print('pred: ', class_names[pred_ind], '(', '%.2f'%pred, ')', ', delta: ', abs(delta))

    add_attributions_to_visualizer(attributions_ig, vocab_itos, text, pred, pred_ind, label, delta, vis_data_records_ig)

def add_attributions_to_visualizer(attributions, vocab_itos, text, pred, pred_ind, label, delta, vis_data_records):
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()
    print(attributions.shape)
    
    class_names = ["Neutral", "Hate"]

    # storing couple samples in an array for visualization purposes
    vis_data_records.append(visualization.VisualizationDataRecord(
                            attributions,
                            pred,
                            class_names[pred_ind],
                            class_names[label],
                            class_names[1],
                            attributions.sum(),
                            text,
                            delta))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", help="unprocessed OLID training dataset", default="data/training_data/offenseval-training-v1.tsv")
    parser.add_argument("--testset_data", help="unprocessed OLID testset dataset", default="data/test_data/testset-levela.tsv")
    parser.add_argument("--test_labels_data", help="unprocessed OLID test labels dataset", default="data/test_data/labels-levela.csv")
    parser.add_argument("--model", help="model to use. Choices are: BasicLSTM, ...", default='BiLSTM')
    parser.add_argument("--batch_size", help="batch size", type=int, default=1)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-3)
    parser.add_argument("--optimizer_type", help="optimizer: adam, sgd", default='adam')
    parser.add_argument("--loss_criterion", help="loss function: bceloss, crossentropy", default='bceloss')
    parser.add_argument("--epochs", default=10, help="cpu or cuda for gpu", type=int)
    parser.add_argument("--patience_es", default=2, help="nb epochs before early stopping", type=int)
    parser.add_argument("--do_save", default=1, help="1 for saving stats and figures, else 0", type=int)
    parser.add_argument("--save_condition", help="save model with"+\
                        " condition on best val_acc (acc) or lowest val_loss(loss)", default='acc')
    parser.add_argument("--device", default='' , help="cpu or cuda for gpu")
    parser.add_argument("--model_path", default='saved-models/BiLSTM_2021-12-03_23-58-08_trained_testAcc=0.5561.pth' , help="saved model to load")


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
    saved_model_path = args.model_path

    if args.device in ['cuda', 'cpu']:
        device = args.device
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("Device:", device)

    field, tokenizer, train_data, val_data, test_data = get_datasets(training_data, testset_data, test_labels_data)

    vocab_stoi, vocab_itos = get_vocab_stoi_itos(field)

    dataloaders = get_dataloaders(train_data, val_data, test_data, batch_size, device)

    model = load_model(model_type,field,device)
    model = load_trained_model(model, saved_model_path, device)

    # lime_explainability(model, vocab_stoi, vocab_itos, dataloaders)



    #
    vis_data_records_ig = batch_model_explainability(model, vocab_stoi, vocab_itos, dataloaders, field, device)
    print(vis_data_records_ig)
    visualization.visualize_text(vis_data_records_ig)
