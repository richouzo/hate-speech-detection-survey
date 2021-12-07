import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BasicLSTM(nn.Module):
    
    def __init__(self, dim_emb = 300, num_words = 10000, hidden_dim = 128, num_layers = 2, output_dim = 1):
        super(BasicLSTM, self).__init__()
        
        self.emb = nn.Embedding(num_words, dim_emb, padding_idx=1 )
        #simple two-layered lstm
        self.lstm = nn.LSTM(dim_emb, hidden_dim, num_layers, batch_first = False)
        #linear layer to get the prediction
        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        #we send our word indices through an embedding layer to turn the into word vectors
        #all our models must to this step, the simplest way is pytorch embedding layer used in this example
        emb = self.emb(x)
       
        #batches are padded during preprocessing we use a mask to identify padding tokens
        #we must let the lstm know which tokens are paddings so that it can ignore them in its calculations, for rnn (grus or lstms) we can reuse the pack_padded_sequence method
        #but for other layers or architectures this step must still be done in another way

        #we create a padded sequence using a mask on the input 
        X_mask = (torch.where(x==1,0,1))
        X_packed = pack_padded_sequence(emb, X_mask.sum(axis=0).tolist(), batch_first=False,enforce_sorted=False)
        #we run the padded sequence through the lstm
        lstm_hidden = self.lstm(X_packed)[1][0][-1]

        #we retrive the last hidden node in the lstm and use it to predict the class using a sigmoid
        #the training uses binary cross entropy so model output should be float value between 0 and 1
        return self.lin(torch.tanh(lstm_hidden)).view(-1)