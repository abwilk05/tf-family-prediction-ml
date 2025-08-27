import numpy as np
import torch

from models.FCN import FCN

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_models(model_name,device,n_fams=20,k=5):
    """
    Loads models from saves
    """
    models = []
    for i in range(1,k+1):
        model = FCN(n_fams).to(device)
        model.load_state_dict(torch.load(f'models/saves/{model_name}/fold{i}.pth'))
        models.append(model)
    return models