import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModel

class DistillBert(nn.Module):
    def __init__(self, freeze_bert=False, dropout=0.2, num_classes=2):
        super(DistillBert, self).__init__()

        pretrained_model_name = "distilbert-base-uncased"

        config = AutoConfig.from_pretrained(
            pretrained_model_name, num_labels=num_classes
        )

        self.bert = AutoModel.from_pretrained(pretrained_model_name, config=config)
        hidden_size = config.hidden_size
        self.linear1 = nn.Linear(hidden_size, hidden_size//2)
        self.linear2 = nn.Linear(hidden_size//2, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Freeze bert layers
        if freeze_bert:
            for name, p in self.named_parameters():
                if "bert" in name:
                    p.requires_grad = False

    def forward(self, x):
        x = x.permute(1, 0)
        attention_mask = (torch.where(x==0, 0, 1)) # pad_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token) = 0
        bert_output = self.bert(input_ids=x, attention_mask=attention_mask)

        seq_output = bert_output[0]  # (bs, seq_len, dim)
        pooled_output = seq_output.mean(axis=1)  # (bs, dim)

        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        out = self.relu(self.linear1(pooled_output))  # (bs, num_classes)
        out = self.linear2(out)  # (bs, num_classes)

        return out

class DistillBertEmotion(nn.Module):
    def __init__(self, freeze_bert=False, dropout=0.2, num_classes=2):
        super(DistillBertEmotion, self).__init__()

        pretrained_model_name = "bhadresh-savani/distilbert-base-uncased-emotion"

        config = AutoConfig.from_pretrained(
            pretrained_model_name, num_labels=num_classes
        )

        self.bert = AutoModel.from_pretrained(pretrained_model_name, config=config)
        hidden_size = config.hidden_size
        self.linear1 = nn.Linear(hidden_size, hidden_size//2)
        self.linear2 = nn.Linear(hidden_size//2, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Freeze bert layers
        if freeze_bert:
            for name, p in self.named_parameters():
                if "bert" in name:
                    p.requires_grad = False

    def forward(self, x):
        x = x.permute(1, 0)
        attention_mask = (torch.where(x==0, 0, 1)) # pad_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token) = 0
        bert_output = self.bert(input_ids=x, attention_mask=attention_mask)

        seq_output = bert_output[0]  # (bs, seq_len, dim)
        pooled_output = seq_output.mean(axis=1)  # (bs, dim)

        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        out = self.relu(self.linear1(pooled_output))  # (bs, num_classes)
        out = self.linear2(out)  # (bs, num_classes)

        return out
