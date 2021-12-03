import time
import numpy as np

import tqdm
import copy

import torch

from sklearn.metrics import f1_score

from utils import EarlyStopping

def train_model(model, criterion, optimizer, dataloaders, history_training, 
                scheduler=None, num_epochs=10, patience_es=5, training_remaining=1):
    '''
    Main training function
    '''
    print("\n**TRAINING**\n")
    # Init Early stoping class
    early_stopping = EarlyStopping(patience=patience_es, verbose=False, delta=0)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    history_training['epochs'] = np.arange(num_epochs)

    # Iterate over epochs.
    for epoch in range(num_epochs):
        lasttime = time.time()
        print('Epoch {}/{} | Trainings remaining: {}'.format(epoch, num_epochs - 1, training_remaining))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            preds_list = []
            labels_list = []

            nb_batches = len(dataloaders[phase])
            length_phase = len(dataloaders[phase].dataset)

            # Iterate over data.
            pbar = tqdm.tqdm([i for i in range(nb_batches)])

            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                pbar.update()
                pbar.set_description("Processing batch %s" % str(batch_idx+1))
                labels = labels.float()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model.forward(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                preds = torch.where(outputs > 0.5, 1, 0).tolist()
                preds_list += preds
                labels_list += labels.tolist()

            pbar.close()
            if phase == 'train' and scheduler != None and epoch != 0:
                scheduler.step(history_training['val_loss'][-1])

            epoch_loss = running_loss / length_phase
            epoch_acc = f1_score(labels_list, preds_list)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))            

            history_training[f'{phase}_loss'].append(epoch_loss)
            history_training[f'{phase}_acc'].append(epoch_acc)

            # deep copy the model
            if phase == 'val':
                valid_loss = epoch_loss # Register validation loss for Early Stopping

                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        print("Epoch complete in {:.1f}s\n".format(time.time() - lasttime))

        # Check Early Stopping
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            history_training['best_val_acc'] = best_val_acc
            history_training['epochs'] = np.arange(epoch + 1)
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    best_val_acc = round(float(best_val_acc), 4)
    print('Best val Acc: {:4f}'.format(best_val_acc))
    history_training['best_val_acc'] = best_val_acc

    # Load best model weights
    model.load_state_dict(best_model_wts)

    return (model, history_training)

def test_model(model, history_training, criterion, dataloaders):
    """
    Testing function. 
    Print the loss and accuracy after the inference on the testset.
    """
    print("\n\n**TESTING**\n")

    start_time = time.time()

    phase = "test"
    model.eval()

    running_loss = 0.0
    preds_list = []
    labels_list = []

    nb_batches = len(dataloaders[phase])
    length_phase = len(dataloaders[phase].dataset)

    pbar = tqdm.tqdm([i for i in range(nb_batches)])

    # Iterate over data.
    for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
        pbar.update()
        pbar.set_description("Processing batch %s" % str(batch_idx+1))
        labels = labels.float()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        preds = torch.where(outputs > 0.5, 1, 0).tolist()
        preds_list += preds
        labels_list += labels.tolist()

    pbar.close()

    test_loss = running_loss / length_phase
    test_acc = f1_score(labels_list, preds_list)
    test_acc = round(float(test_acc), 4)
    history_training['test_acc'] = test_acc

    history_training['y_pred'] = preds_list
    history_training['y_true'] = labels_list

    print('\nTest stats -  Loss: {:.4f} Acc: {:.2f}%'.format(test_loss, test_acc*100))
    print("Inference on Testset complete in {:.1f}s\n".format(time.time() - start_time))

    return history_training
