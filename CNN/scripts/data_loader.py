import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np

class CustomDataset(Dataset):
    """
    Contains PWM and class label
    """
    
    def __init__(self,data,fam):
        self.data = data
        self.fam = fam

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        sample = torch.tensor(self.data[index], dtype=torch.float).to(device)
        label = torch.tensor(self.fam[index], dtype=torch.long).to(device)

        return sample, label
    
def get_k_folds(pwms,fams,k=5):
    """
    Gets indices for k folds
    """
    kf = StratifiedKFold(n_splits=k,shuffle=True,random_state=42)
    k_folds = []
    for train_index, val_index in kf.split(pwms,fams):
        k_folds.append([train_index,val_index])
    return k_folds

def prep_data(data,labels,train_index,val_index):
    """
    Prepares data from specific fold for training
    """
    
    train_data = [data[i] for i in train_index]
    train_labels = [labels[i] for i in train_index]
    val_data = [data[i] for i in val_index]
    val_labels = [labels[i] for i in val_index]

    # Pytorch data structures
    batch_size = 1
    train_dataset = CustomDataset(train_data, train_labels)
    val_dataset = CustomDataset(val_data, val_labels)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return trainloader, valloader

def consensus(pwms):
    """
    Returns the "consensus motifs" for a list of pwm-type motifs.
    """
    rng = np.random.default_rng(5)
    consensus_motifs = []
    for mat in pwms:
        onehot = np.zeros_like(mat, dtype=int)
        for j in range(mat.shape[1]):
            col = mat[:, j]
            max_val = col.max()
            candidates = np.flatnonzero(col == max_val)
            choice = rng.choice(candidates)
            onehot[choice, j] = 1
        consensus_motifs.append(onehot)
    return consensus_motifs