from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import time
import pickle

from scripts.data_loader import prep_data
from models.FCN import FCN

def train_CV(data, labels, k_folds, device, model_name, n_fams=20):
    """
    Trains k models using k-fold cross validation
    """
    models = []
    
    writer = SummaryWriter(f'runs/{model_name}')
    
    for fold, [train_index,val_index] in enumerate(k_folds):

        model = FCN(n_fams).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
        # optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_func = nn.CrossEntropyLoss()
        
        # prep data
        trainloader, valloader = prep_data(data,labels,train_index,val_index)
        # perform training
        model_trained = train(trainloader,valloader,fold+1,model,optimizer,loss_func,writer,model_name)
        models.append(model_trained)

    writer.close()

    return models
        
def train(trainloader, testloader, fold, model, optimizer, loss_func, writer, model_name):
    """
    Trains an independent fold in k-fold CV.
    """

    # Early stopping criteria
    patience = 100 # 7
    best_val_loss = float('inf')
    counter = 0 

    # Tracking performance across epochs
    train_losses = []
    val_losses = []
    
    num_epochs = 100
    for epoch in range(num_epochs):
        
        # Training Data
        model.train()
        train_loss = []
        for input, label in trainloader:
            optimizer.zero_grad()
            output = model(input)
            loss = loss_func(output, label)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)
        train_losses.append(train_loss)

        # Val Data
        model.eval()
        val_loss = []
        with torch.no_grad():
            for input, label in testloader:
                output = model(input)
                loss = loss_func(output, label)
                val_loss.append(loss.item())
        val_loss = np.mean(val_loss)
        val_losses.append(val_loss)

        writer.add_scalars(
            f"Fold_{fold}",
            {
                "Train": train_loss,
                "Val":   val_loss
            },
            epoch
        )
    
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            print(f"Early stopping after {epoch + 1} epochs.")
            break
    
    torch.save(model.state_dict(), f'models/saves/{model_name}/fold{fold}.pth')

    return model
        
def get_y_pred(data, labels, models, k_folds, model_name, save=False):

    start = time.time()
    
    y_pred = ['']*len(labels)

    for i, [train_index,val_index] in enumerate(k_folds):
        model = models[i]
        trainloader, valloader = prep_data(data,labels,train_index,val_index)
        for i, (input,label) in enumerate(valloader):
            output = model(input)
            prob, pred = torch.max(output, 1)
            y_pred[val_index[i]] = pred.item()
    
    end = time.time()

    print(model_name, end-start)

    if save:
        with open(f'data/y_pred/{model_name}.pkl', 'wb') as f:
            pickle.dump(y_pred, f)

def accuracy_n_f1(data, labels, models, k_folds, model_name):

    accuracy = []
    f1 = []
    for i, [train_index,val_index] in enumerate(k_folds):
        model = models[i]
        trainloader, valloader = prep_data(data,labels,train_index,val_index)
        y_true = []
        y_pred = []
        for i, (input,label) in enumerate(valloader):
            output = model(input)
            prob, pred = torch.max(output, 1)
            y_pred.append(pred.item())
            y_true.append(label.item())
        accuracy.append(accuracy_score(y_true,y_pred))
        f1.append(f1_score(y_true,y_pred,average='macro'))
    
    print(model_name)
    print(f'Accuracy: {accuracy}')
    print(f'F1: {f1}')

def per_class_acc_f1(y_true, y_pred,model_name):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    accuracy = []
    
    classes = np.unique(y_true)
    
    for cls in classes:
        mask = (y_true == cls)
        if np.sum(mask) > 0:
            accuracy_cls = np.sum(y_pred[mask] == cls) / np.sum(mask)
        else:
            accuracy_cls = np.nan
        accuracy.append(accuracy_cls)
    
    f1_scores = f1_score(y_true, y_pred, average=None, labels=range(0,20))

    print(f'{model_name}')
    print('Accuracy:',accuracy)
    print('F1:',list(f1_scores))