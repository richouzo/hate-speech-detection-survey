import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class PyramidCNN(nn.Module):
    
    def __init__(self, dim_emb = 300, num_words = 10000, hidden_dim = 256, context_size = 1, alpha=1):
        super(PyramidCNN, self).__init__()
        
        self.alpha = alpha
        self.emb = [nn.Embedding(num_words, dim_emb, padding_idx=1).cuda()]
        for i in range(context_size):
          self.emb.append(nn.Embedding(num_words, dim_emb, padding_idx=1).cuda())
          self.emb.append(nn.Embedding(num_words, dim_emb, padding_idx=1).cuda())

        self.conv1 = nn.Conv1d(dim_emb, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim/2, 3, padding=1)
        
        self.lstm = nn.LSTM(hidden_dim/2, hidden_dim, num_layers, batch_first = False)
        
        self.linear1 = nn.Linear(hidden_dim, hidden_dim/2)
        self.relu1  = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim/2, hidden_dim/2)
        self.relu2 = nn.ReLU() 
        self.linear3 = nn.Linear(hidden_dim/2, 1)


    def forward(self, x):

        emb = self.emb[0](x)
        x_prev = x_next = x
        for i in range(self.context_size):
          x_prev = torch.roll(x,1,dims=1)
          x_prev[:,0] = 1
          x_next = torch.roll(x,-1,dims=1)
          x_next[:,-1] = 1
          emb = emb + self.alpha*(self.emb[2*i+1](x_prev) + self.emb[2*i+2](x_next))

        emb = emb.permute(1, 2, 0)

        conv1 = F.relu(self.conv1(emb))
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