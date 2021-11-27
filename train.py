import numpy as np
import tqdm
import torch

from sklearn.metrics import f1_score
from torch import optim
import torch.nn as nn


def train(model, train_iterator, test_iterator, num_epochs=10):
    
    ct = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("start training...")
    for epoch in tqdm.tqdm(range(num_epochs)):
        
        
        with tqdm.tqdm(
                train_iterator,
                desc="epoch {}".format(epoch + 1),
                unit="batch",
                total=len(train_iterator)) as batch_iterator:
            model.train()
            total_loss = 0.0
            
            for i, batch in enumerate(batch_iterator, start=1):

                X = batch.text
                y = batch.label.float()

                model.zero_grad()

                preds = model.forward(X)
                loss = ct(preds,y)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

                batch_iterator.set_postfix(mean_loss=total_loss / i, current_loss=loss.item())
            
        train_preds = []
        train_y = []
        test_preds = []
        test_y = []

        for batch in train_iterator:
            preds = model.forward(batch.text)
            preds = torch.where(preds >0.5, 1, 0).tolist()
            train_preds += preds 
            train_y += batch.label.tolist()

        for batch in test_iterator:
            preds = model.forward(batch.text)
            preds = torch.where(preds >0.5, 1, 0).tolist()
            test_preds += preds 
            test_y += batch.label.tolist()


        train_acc = f1_score(train_y, train_preds)
        print(len(train_y))
        print(len(test_y))
        test_acc = f1_score(test_y, test_preds)  
        print('epoch loss:',total_loss)
        print('epoch train f1:',train_acc)
        print('epoch test f1:',test_acc)