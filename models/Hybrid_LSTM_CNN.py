import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class HybridLSTMCNN(nn.Module):
    
    def __init__(self, fix_length, dim_emb = 100, num_words = 10000, hidden_dim = 128, num_layers = 2):
        super(HybridLSTMCNN, self).__init__()
        
        self.emb = nn.Embedding(num_words, dim_emb, padding_idx=1)       
        self.lstm = nn.LSTM(100, 500, num_layers, batch_first = False)
        
        self.conv1 = nn.Conv1d(500, 512, 3, padding=1)
        self.conv2 = nn.Conv1d(512, 256, 3, padding=1)
        self.conv3 = nn.Conv1d(256, 128, 3, padding=1)
        
        self.flatten = nn.Flatten()
        
        self.linear1 = nn.Linear(128*fix_length, 128)
        self.relu1  = nn.ReLU()
        self.linear2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU() 
        self.linear3 = nn.Linear(128, 1)
        

    def forward(self, x):
        emb = self.emb(x)
        
        X_mask = (torch.where(x==1,0,1))
        X_packed = pack_padded_sequence(emb, X_mask.sum(axis=0).tolist(), batch_first=False, enforce_sorted=False)
        lstm_output, _ = self.lstm(X_packed)
        X_pad, _ = pad_packed_sequence(lstm_output, batch_first=False, total_length = x.shape[0])
        
        X_reshaped = X_pad.permute(1, 2, 0)
        conv1 = F.relu(self.conv1(X_reshaped))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = self.conv3(conv2)

        
        flattened = self.flatten(conv3)
        linear1 = self.relu1(self.linear1(flattened))
        linear2 = self.relu2(self.linear2(linear1))
        linear3 = self.linear3(linear2)
        
        return linear3.view(-1)