import numpy as np
import tqdm
import torch
from torchtext.legacy.data import Field, LabelField, TabularDataset, BucketIterator
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset
import spacy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch import optim
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from preprocess_utils import *
from train import train
from models import BasicLSTM
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--data", help="unprocessed OLID dataset")
parser.add_argument("--model", help="model to use. Choices are : BasicLSTM, ", default=None)
parser.add_argument("--batch_size", help="batch size",type=int, default=128)
parser.add_argument("--device", default='cpu' , help="cpu or cuda for gpu")
parser.add_argument("--epochs", default=10 , help="cpu or cuda for gpu", type=int)

args = parser.parse_args()

olid_file = args.data
batch_size = args.batch_size
dev = args.device
spacy_en = spacy.load("en_core_web_sm")

# preprocessing of the tweets
tweets, classes = format_file(olid_file)
print("file loaded and formatted..")
train_test_split_tocsv(tweets, classes, test_size=0.2)
print("data split into train/test")


def tokenizer(text):    
    return [tok.text for tok in spacy_en.tokenizer(text)]

ENGLISH, LABEL, train_data, test_data = create_fields_dataset(tokenizer)

# build vocabularies using training set
print("fields and dataset object created")
ENGLISH.build_vocab(train_data, max_size=10000, min_freq=2)
LABEL.build_vocab(train_data)

print("vocabulary built..")
train_iterator, test_iterator = create_iterators(train_data, test_data, batch_size, dev)
print("iterators created..")
#instanciate model (all models need to be added here)
if args.model == 'MORE MODELS NAMES':
    HateNet = 0 #other models here
else:
    HateNet = BasicLSTM.BasicLSTM(dim_emb = 300, num_words = ENGLISH.vocab.__len__(), hidden_dim = 128, num_layers = 2, output_dim = 1)
    if dev=='cuda':
       HateNet.cuda()


print("model loaded ..")
print('training starts')
train(HateNet, train_iterator, test_iterator, num_epochs=args.epochs)

