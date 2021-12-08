import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import AutoModelForSequenceClassification

class AutoTransformer(nn.Module):

    def __init__(self, dim_emb=768, num_words = 10000, hidden_dim = 128, num_layers = 2, output_dim = 1, hidden_dropout_prob = 0.5):
        super(AutoTransformer, self).__init__()

        self.emb = nn.Embedding(num_words, dim_emb, padding_idx=1)
        self.bert = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", 
                                                                       output_hidden_states=True) 
                                                                       #"bert-base-cased-finetuned-mrpc")
                                                                       # "distilbert-base-uncased-finetuned-sst-2-english"
                                                                       
        self.rnn = nn.LSTM(dim_emb, hidden_dim, num_layers, batch_first = False ,bidirectional=True)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.Tanh()
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x):

        #we send our word indices through an embedding layer to turn the into word vectors
        #all our models must to this step, the simplest way is pytorch embedding layer used in this example

        output = self.bert(x.permute(1, 0)).hidden_states[0]
        output = output.permute(1, 0, 2)
    
        X_mask = (torch.where(x==1,0,1))
    
        X_packed = pack_padded_sequence(output, X_mask.sum(axis=0).tolist(), 
                                        batch_first=False, enforce_sorted=False)
        rnn_hidden = self.rnn(X_packed)[1][0][-1]

        logits = self.linear(torch.tanh(rnn_hidden))

        logits = self.act(logits)
        #print(logits.shape)
        logits = self.dropout(logits)
        logits = self.classifier(logits)
        return logits.view(-1)

