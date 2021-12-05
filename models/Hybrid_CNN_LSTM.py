import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class HybridCNNLSTM(nn.Module):
    
    def __init__(self, dim_emb = 300, num_words = 10000, hidden_dim = 128, num_layers = 2):
        super(HybridCNNLSTM, self).__init__()
        
        self.emb = nn.Embedding(num_words, dim_emb, padding_idx=1)
        
        self.conv1 = nn.Conv1d(dim_emb, 512, 3, padding=1)
        self.conv2 = nn.Conv1d(512, 256, 3, padding=1)
        self.conv3 = nn.Conv1d(256, 128, 3, padding=1)
        
        self.lstm = nn.LSTM(128, 500, num_layers, batch_first = False)
        
        self.linear1 = nn.Linear(500, 128)
        self.relu1  = nn.ReLU()
        self.linear2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU() 
        self.linear3 = nn.Linear(128, 1)


    def forward(self, x):
        emb = self.emb(x)
        emb_reshaped = emb.permute(1, 2, 0)

        conv1 = F.relu(self.conv1(emb_reshaped))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = self.conv3(conv2)
        conv3_reshaped = conv3.permute(2, 0, 1)

        X_mask = (torch.where(x==1,0,1))

        X_packed = pack_padded_sequence(conv3_reshaped, X_mask.sum(axis=0).tolist(), batch_first=False, enforce_sorted=False)
        lstm_hidden, _ = self.lstm(X_packed)[1][-1]

        linear1 = self.relu1(self.linear1(lstm_hidden))
        linear2 = self.relu2(self.linear2(linear1))
        linear3 = self.linear3(linear2)

        return linear3.view(-1)