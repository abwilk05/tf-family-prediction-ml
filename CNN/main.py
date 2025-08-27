import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

from scripts.get_pwms import save_pwms, load_pwms, preprocess
from scripts.helper import set_seed
from scripts.data_loader import get_k_folds, consensus
from scripts.train_eval import train_CV, get_y_pred, accuracy_n_f1, per_class_acc_f1
from scripts.helper import load_models
from scripts.plots import total_accuracy, total_class_accuracy, total_conf_matrix, total_confidence, IG, confidence, class_accuracy_comp, class_dstb
from scripts.plots_new import accuracy, time, class_accuracy, size_acc_corr

def main():
    
    # ### Initialize seed, device, model_name ###
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = '03_26_10fold'

    # ### Load data ###
    motifs = load_pwms()
    pwms, fams, m_ids, id_to_fam, id_to_fam_cnts, n_fams = preprocess(motifs, print_info=False)
    k_folds = get_k_folds(pwms,fams,k=10)

    ### Train Model ###
    # Make new directory: models/saves/{model_name}
    # models = train_CV(pwms,fams,k_folds,device,model_name)

    ### Load Model ###
    models = load_models(model_name,device,k=10)

    ### Evaluation ###
    # Use functions as needed

if __name__ == '__main__':
    main()