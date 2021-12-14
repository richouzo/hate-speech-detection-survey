import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class PyramidCNN(nn.Module):
    
    def __init__(self, dim_emb = 300, num_words = 10000, hidden_dim = 256, context_size = 1, alpha=0.2, 
        fcs=[128], batch_norm=0, pyramid=[128,256,256], pad_len=32, pooling_size=2, glove=None):
        super(PyramidCNN, self).__init__()
        
        self.context_size = context_size
        self.alpha = alpha 
        self.pad_len = pad_len
        self.num_blocks = len(pyramid)
        self.batch_norm = batch_norm

        self.bn = nn.BatchNorm1d(dim_emb)

        if glove == None:
            self.emb = [nn.Embedding(num_words, dim_emb, padding_idx=1)]
            for i in range(context_size):
              self.emb.append(nn.Embedding(num_words, dim_emb, padding_idx=1))
              self.emb.append(nn.Embedding(num_words, dim_emb, padding_idx=1))
            self.emb = nn.ModuleList(self.emb) 
        else:
            self.emb = [nn.Embedding.from_pretrained(glove, freeze=False, padding_idx=1)]
            for i in range(context_size):
              self.emb.append(nn.Embedding.from_pretrained(glove, freeze=False, padding_idx=1))
              self.emb.append(nn.Embedding.from_pretrained(glove, freeze=False, padding_idx=1))
            self.emb = nn.ModuleList(self.emb) 

        self.convs = []
        self.dos = []
        self.rel = nn.ReLU()
        self.pool = nn.MaxPool1d(pooling_size)
        for i in range(self.num_blocks):
            self.convs.append(nn.Conv1d(dim_emb, pyramid[i], 3, padding='same'))
            self.convs.append(nn.Conv1d(pyramid[i], dim_emb, 3, padding='same'))
            self.dos.append(nn.Dropout(p=0.5))
        self.convs = nn.ModuleList(self.convs)
        self.dos = nn.ModuleList(self.dos)
        
        self.fcs = [nn.Linear(dim_emb, fcs[0])]
        self.fc_rel = nn.LeakyReLU()
        for i in range(len(fcs)-1):
            self.fcs.append(nn.Linear(fcs[i], fcs[i+1]))
        self.flin = nn.Linear(fcs[-1],2)
        self.fcs = nn.ModuleList(self.fcs)


    def forward(self, x):
        
        min_len=self.pad_len
        
        if x.shape[0]<min_len:
            pad_len = min_len - x.shape[0]
            pad = torch.ones((pad_len,x.shape[1]), dtype=torch.int32, device=x.device)
            x = torch.cat([x,pad],axis=0)
        emb = self.emb[0](x)
 
        for i in range(self.context_size):
          x_prev = torch.roll(x,1,dims=1)
          x_prev[:,0] = 1
          x_next = torch.roll(x,-1,dims=1)
          x_next[:,-1] = 1
          emb = emb + self.alpha*(self.emb[2*i+1](x_prev) + self.emb[2*i+2](x_next))
        emb = emb.permute(1, 2, 0)

        emb = self.bn(emb)
        
        res = emb
        for i in range(self.num_blocks):
            conv1 = self.rel(self.convs[2*i](res))
            conv2 = self.rel(self.dos[i](self.convs[2*i+1](conv1)))
            pool = self.pool(conv2)
        
            res = pool + self.rel(self.pool(res))

        output = res.permute(2, 0, 1).mean(axis=0)

        fc = output
        for lin in self.fcs:
            fc = self.fc_rel(lin(fc))

        return self.flin(fc)