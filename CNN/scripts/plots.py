import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader
import torch.nn.functional as F
import logomaker as lm
from scipy.stats import entropy
import matplotlib.ticker as ticker
from scipy import stats
from statannotations.Annotator import Annotator
import pickle

from scripts.data_loader import CustomDataset,prep_data

def total_accuracy(data, labels, models, k_folds, model_name, save=False):
    """
    Plots total accuracy across k folds/models
    """

    # Obtain accuracies
    accuracies = []
    for i, [train_index,val_index] in enumerate(k_folds):
        model = models[i]
        trainloader, valloader = prep_data(data,labels,train_index,val_index)
        accuracies.append(accuracy(model,valloader))

    # Plot Accuracies
    plt.figure(figsize=(4,6))
    plt.boxplot(accuracies)
    plt.ylim(0.5,1)
    plt.ylabel('Accuracy')
    plt.xticks([])
    plt.title(f'{model_name} (Accuracy={np.mean(accuracies)*100:.2f}%)')
    if not save:
        plt.show()
    else:
        plt.savefig(f'outputs/{model_name}/accuracy.png')
        plt.close()

def accuracy(model,val_loader):
    """
    Computes accuracy for one model/fold
    """
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for input,label in val_loader:
            output = model(input)
            prob, pred = torch.max(output, 1)
            
            total += 1
            if pred == label:
                correct += 1

    return correct/total

def total_class_accuracy(data, labels, models, k_folds, model_name, id_to_fam_cnts, save=False, n_fams=20):
    """
    Plots total per class accuracy across k folds/models
    """

    df = pd.DataFrame(columns=range(0,n_fams))

    for i, [train_index,val_index] in enumerate(k_folds):
        model = models[i]
        trainloader, valloader = prep_data(data,labels,train_index,val_index)
        new_row = class_accuracy(model, valloader, n_fams)
        df.loc[i] = new_row

    df = df.rename(columns=id_to_fam_cnts) # change column names
    df = df[df.columns[::-1]] # reverse columns for plotting
    
    ax = df.boxplot(vert=False, patch_artist=True, figsize=(10, 6))
    boxes = ax.artists  # Each box is a Rectangle in this case
    for b in boxes:
        b.set_facecolor("lightgray")

    plt.title(f'{model_name}')
    plt.xlabel("Accuracy")
    plt.ylabel("TF Family")
    plt.tight_layout()
    if not save:
        plt.show()
    else:
        plt.savefig(f'outputs/{model_name}/accuracy_class.png')
        plt.close()

def class_accuracy(model, val_loader, n_fams=20):
    """
    Computes per class accuracy for one model/fold
    """
    model.eval()
    correct = {i : 0 for i in range(0,n_fams)}
    total = {i : 0 for i in range(0,n_fams)}
    with torch.no_grad():
        for input,label in val_loader:
            output = model(input)
            prob, pred = torch.max(output, 1)

            total[label.item()] += 1
            if pred == label:
                correct[label.item()] += 1

    return {k: correct[k]/total[k] for k in range(0,n_fams)}

def total_conf_matrix(data, labels, models, k_folds, model_name, id_to_fam, save=False, n_fams=20):
    
    cm = np.zeros((20, 20), dtype=float)
    for i, [train_index,val_index] in enumerate(k_folds):
        model = models[i]
        trainloader, valloader = prep_data(data,labels,train_index,val_index)
        cm += conf_matrix(model, valloader)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm = cm / row_sums   

    classes = id_to_fam.values()
    plt.figure(figsize=(12,10))
    plt.title(f'{model_name}')
    sns.heatmap(cm*100, annot=True, fmt='.1f', cmap="Blues", cbar=False)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    yticks = np.array(range(len(classes))) + 0.5
    plt.yticks(yticks, classes, rotation=0)
    plt.tight_layout()
    if not save:
        plt.show()
    else:
        plt.savefig(f'outputs/{model_name}/conf_matrix.png')
        plt.close()

def conf_matrix(model, val_loader):
    """
    Makes multiclass confusion matrix for one fold
    """
    true = []
    preds = []
    model.eval()
    with torch.no_grad():
        for input, label in val_loader:            
            output = model(input)
            prob, pred = torch.max(output, 1)
            true.append(label.item())
            preds.append(pred.item())
    return confusion_matrix(true, preds)

def total_confidence(data, labels, models, k_folds, model_name, save=False, n_fams=20):
    
    correct = []
    incorrect = []

    for i, [train_index,val_index] in enumerate(k_folds):
        model = models[i]
        trainloader, valloader = prep_data(data,labels,train_index,val_index)
        model.eval()
        with torch.no_grad():
            for input,label in valloader:
                output = model(input)
                _, pred = torch.max(output, 1)
                softmax = F.softmax(output, dim=1).tolist()[0]
                prob = sorted(softmax)[-1]

                if pred == label:
                    correct.append(prob)
                else:
                    incorrect.append(prob)

    data = pd.DataFrame({
        'Confidence': correct + incorrect,
        'Group': ['Correct'] * len(correct) + ['Incorrect'] * len(incorrect)
    })

    plt.figure(figsize=(8,9))
    ax = sns.boxplot(x='Group', y='Confidence', data=data, palette=['green', 'red'])

    ax.set_xlabel("")
    ax.set_xticklabels([f'Correct (n={len(correct)})', f'Incorrect (n={len(incorrect)})'], rotation=0)
    ax.set_ylim(0,1.1)

    pairs = [("Correct", "Incorrect")]

    annotator = Annotator(ax, pairs, data=data, x='Group', y='Confidence')
    annotator.configure(test='Mann-Whitney', text_format='star',loc='inside')
    annotator.apply_and_annotate()
    
    if not save:
        plt.show()
    else:
        plt.savefig(f'outputs/{model_name}/confidence.png')
        plt.close()

def confidence(m_id, which_pred, data, labels, m_ids, models, k_folds, device):
    """
    Gives confidence measure of prediction for m_id. which_pred specifies which prediction 1 or 2...
    """
    # Find index and fold
    idx = m_ids.index(m_id)
    k=0 # fold location
    for [train_index,val_index] in k_folds:
        if idx in val_index:
            break
        k += 1

    model = models[k]
    input = torch.tensor(data[idx], dtype=torch.float).to(device)
    input = input.unsqueeze(0)
    label = torch.tensor(labels[idx], dtype=torch.long).to(device)
    output = model(input)
    _, pred = torch.max(output, 1)
    softmax = F.softmax(output, dim=1).tolist()[0]
    prob = sorted(softmax)[-which_pred]
    fam = softmax.index(prob)

    correct = (pred == label).item()

    return correct, prob, fam

def IG(m_id, data, labels, m_ids, models, k_folds, model_name, id_to_fam, device, save=False, n_fams=20):   
    """
    Plot integrated gradients feature importance plot for a given motif, specified by m_id
    """
    # Find index and fold
    idx = m_ids.index(m_id)
    k=0 # fold location
    for [train_index,val_index] in k_folds:
        if idx in val_index:
            break
        k += 1

    # Preparing data
    model = models[k]
    input = torch.tensor(data[idx], dtype=torch.float).to(device)
    input = input.unsqueeze(0)
    label = torch.tensor(labels[idx], dtype=torch.long).to(device)
    output = model(input)
    baseline = torch.full(input.shape, 0.25).to(device)

    # Integrated gradients to get importance vector
    IG = IntegratedGradients(model)
    importance = IG.attribute(input, baseline, target=label)
    importance_sum = importance.sum(dim=1)
    importance_scaled = (importance_sum - importance_sum.min()) / (importance_sum.max() - importance_sum.min())
    importance_scaled = importance_scaled.cpu().numpy()

    # The plot
    fig, ax = plt.subplots(2, 1, figsize=(8, 4.5), gridspec_kw={'height_ratios': [2,1]}, sharex=True, constrained_layout=True)
    pwm = input.squeeze().cpu().numpy()
    pwm = np.rot90(pwm, k=-1)
    make_importance_logo(m_id,pwm,importance_scaled,ax)

    # plt.suptitle(f'{m_id}\nTrue: {id_to_fam[label.item()]}\nPredicted: {id_to_fam[output.item()]}')
    plt.suptitle(f'{m_id}')

    if not save:
        plt.show()
    else:
        plt.savefig(f'outputs/{model_name}/importance/bHLH/{m_id}.png')
        plt.close()

def make_importance_logo(m_id,gt,importance,ax):
    """
    Displays PWM with corresponding importance vector
    """
    e = entropy(gt,[0.25,0.25,0.25,0.25], axis=1, base=2)
    gt = gt * e[:,np.newaxis]
    
    for pos in range(gt.shape[1]):
        gt_dict = {'A': [],
            'C':[],
            'G':[],
            'T':[]
            }
    for item in gt:
        gt_dict['A'].append(item[3])
        gt_dict['C'].append(item[2])
        gt_dict['G'].append(item[1])
        gt_dict['T'].append(item[0])

    gt_df = pd.DataFrame(gt_dict)
    
    logo = lm.Logo(gt_df, color_scheme='classic', ax=ax[0], show_spines=False, alpha=0.7, fade_probabilities=False, font_name='DejaVu Sans')

    ax[0].set_ylim(0, 2)
    ax[0].set_ylabel('Bits')
    ax[0].set_xticks([])

    # Heat map importance
    heatmap = ax[1].imshow(importance, cmap='gray_r', aspect='auto')
    ax[1].set_yticks([])  # Hide y-ticks for the heatmap
    ax[1].set_xlabel('Importance')
    cbar = plt.colorbar(heatmap, ax=ax[1], orientation='horizontal', pad=0.3, fraction=0.3)


def class_accuracy_comp(id_to_fam_cnts,save=False):
    
    fcn = [88.5,71.2,76.1,93.0,91.5,90.3,87.5,84.2,80.9,82.8,62.5,75.6,86.8,83.8,80.0,96.2,69.2,63.6,50.0,60.0]
    knn = [96.3,64.7,89.4,91.0,88.5,98.2,91.1,84.2,94.7,86.2,68.7,80.5,47.4,0,93.3,100,80.8,95.5,80,80]

    names = list(id_to_fam_cnts.values())[::-1]
    x = np.arange(len(names))

    plt.figure(figsize=(16, 6))

    plt.scatter(fcn[::-1], x, color='purple', label='Deep Learning', s=100)
    plt.scatter(knn[::-1], x, color='gray', label='KNN Ensemble', s=100)

    plt.yticks(x, names)

    plt.ylabel("TF Family")
    plt.xlabel("Accuracy (%)")
    plt.legend()
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.FixedLocator(x))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    
    plt.grid(which='major', axis='both', linestyle='--', alpha=0.7)
    
    
    plt.tight_layout() 
    if not save:
        plt.show()
    else:
        plt.savefig(f'outputs/03_26_20fold/accuracy_comp.png')
        plt.close()

def class_dstb(fams, id_to_fam):
    from collections import Counter
    
    class_counts = Counter(fams)
    sorted_items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    classes, counts = zip(*sorted_items)
    y_positions = range(len(classes))
    plt.figure(figsize=(7, 6))
    plt.barh(y_positions, counts, color='#ADD8E6')  # Light blue color
    fam_names = [id_to_fam[fam] for fam in classes]
    plt.yticks(y_positions, fam_names, fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()