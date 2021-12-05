import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import AutoModelForSequenceClassification, DistilBertModel
from transformers import AutoConfig, AutoModel

class DistillBert(nn.Module):
    def __init__(self, dropout=0.2, num_classes=2, output_dim=1):
        super(DistillBert, self).__init__()

        pretrained_model_name = "distilbert-base-uncased"

        config = AutoConfig.from_pretrained(
            pretrained_model_name, num_labels=num_classes
        )

        self.bert = AutoModel.from_pretrained(pretrained_model_name, config=config)
        self.classifier = nn.Linear(config.hidden_size, output_dim)
        self.dropout = nn.Dropout(dropout)

        # Freeze bert layers
        for name, p in self.named_parameters():
            if "bert" in name:
                p.requires_grad = False

    def forward(self, x):
        x = x.permute(1, 0)
        # print('x', x)
        attention_mask = (torch.where(x==0, 0, 1)) # pad_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token) = 0
        # print('attention_mask', attention_mask)
        bert_output = self.bert(input_ids=x, attention_mask=attention_mask)
        # print('bert_output', bert_output)
        # qzdqz
        # we only need the hidden state here and don't need
        # transformer output, so index 0
        seq_output = bert_output[0]  # (bs, seq_len, dim)
        # mean pooling, i.e. getting average representation of all tokens
        pooled_output = seq_output.mean(axis=1)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        scores = self.classifier(pooled_output).view(-1)  # (bs, num_classes)

        # print('scores', scores.shape)

        return scores

