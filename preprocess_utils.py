import re
import emoji
import numpy as np

import torch
from torchtext.legacy.data import Field, LabelField, TabularDataset, BucketIterator
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset

import spacy
import pandas as pd

from sklearn.model_selection import train_test_split

spacy_en = spacy.load("en_core_web_sm")
def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def format_training_file(text_file):
    tweets = []
    classes = []

    for line in open(text_file,'r',encoding='utf-8'):
        line = re.sub(r'#([^ ]*)', r'\1', line)
        line = re.sub(r'https.*[^ ]', 'URL', line)
        line = re.sub(r'http.*[^ ]', 'URL', line)
        line = emoji.demojize(line)
        line = re.sub(r'(:.*?:)', r' \1 ', line)
        line = re.sub(' +', ' ', line)
        line = line.rstrip('\n').split('\t')
        tweets.append(line[1])
        classes.append(int(line[2]=='OFF'))

    return tweets[1:], classes[1:]

def train_val_split_tocsv(tweets, classes, val_size=0.2):
    tweets_train, tweets_val, y_train, y_val = train_test_split(tweets, classes, test_size=val_size, random_state=42)

    df_train = pd.DataFrame({'text': tweets_train, 'label': y_train})
    df_val = pd.DataFrame({'text': tweets_val, 'label': y_val})

    df_train.to_csv('data/offenseval_train.csv', index=False)
    df_val.to_csv('data/offenseval_val.csv', index=False)

def format_test_file(text_file_testset, text_file_labels):
    tweets_test = []
    y_test = []
    for line in open(text_file_testset,'r',encoding='utf-8'):
        line = re.sub(r'#([^ ]*)', r'\1', line)
        line = re.sub(r'https.*[^ ]', 'URL', line)
        line = re.sub(r'http.*[^ ]', 'URL', line)
        line = emoji.demojize(line)
        line = re.sub(r'(:.*?:)', r' \1 ', line)
        line = re.sub(' +', ' ', line)
        line = line.rstrip('\n').split('\t')
        tweets_test.append(line[1])
    for line in open(text_file_labels,'r',encoding='utf-8'):
        line = line.rstrip('\n').split('\t')
        y_test.append(int(line[0][-3:]=='OFF'))

    return tweets_test[1:], y_test

def test_tocsv(tweets_test, y_test):
    df_test = pd.DataFrame({'text': tweets_test, 'label': y_test})
    df_test.to_csv('data/offenseval_test.csv', index=False)

def create_fields_dataset(tokenizer_func):
    field = Field(sequential = True, use_vocab = True, tokenize=tokenizer_func, lower=True)
    label = LabelField(dtype=torch.long, batch_first=True, sequential=False)
    fields = [('text', field), ('label', label)]
    print("field objects created")
    train_data, val_data = TabularDataset.splits(
        path = '',
        train='data/offenseval_train.csv',
        test='data/offenseval_val.csv',
        format='csv',
        fields=fields,
        skip_header=True,
    )
    _, test_data = TabularDataset.splits(
        path = '',
        train='data/offenseval_train.csv',
        test='data/offenseval_test.csv',
        format='csv',
        fields=fields,
        skip_header=True,
    )

    return (field, label, train_data, val_data, test_data)

#Create train and test iterators to use during the training loop
def create_iterators(train_data, test_data, batch_size, dev, shuffle=False):
    train_iterator, test_iterator = BucketIterator.splits(
        (train_data, test_data),
        shuffle=shuffle,
        device=dev,
        batch_size=batch_size,
        sort = False,
        )
    return train_iterator, test_iterator

def get_vocab_stoi_itos(field):
    vocab_stoi = field.vocab.stoi
    vocab_itos = field.vocab.itos
    return (vocab_stoi, vocab_itos)

def get_datasets(training_data, testset_data, test_labels_data):
    # preprocessing of the train/validation tweets, then test tweets
    tweets, classes = format_training_file(training_data)
    tweets_test, y_test = format_test_file(testset_data, test_labels_data)
    print("file loaded and formatted..")
    train_val_split_tocsv(tweets, classes, val_size=0.2)
    test_tocsv(tweets_test, y_test)
    print("data split into train/val/test")

    field, label, train_data, val_data, test_data = create_fields_dataset(tokenizer)

    # build vocabularies using training set
    print("fields and dataset object created")
    field.build_vocab(train_data, max_size=10000, min_freq=2)
    label.build_vocab(train_data)
    print("vocabulary built..")

    return (field, train_data, val_data, test_data)

def get_dataloaders(train_data, val_data, test_data, batch_size, device):
    train_iterator, val_iterator = create_iterators(train_data, val_data, batch_size, device, shuffle=True)
    _, test_iterator = create_iterators(train_data, test_data, batch_size, device, shuffle=False)
    print("dataloaders created..")

    dataloaders = {}
    dataloaders['train'] = train_iterator
    dataloaders['val'] = val_iterator
    dataloaders['test'] = test_iterator

    return dataloaders
