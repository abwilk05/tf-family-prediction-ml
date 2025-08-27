import random
import numpy as np
import pandas as pd
import torch
from pyjaspar import jaspardb
from sklearn.metrics import accuracy_score, f1_score
import pickle
from collections import Counter

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_tsv(fpath):
    return pd.read_csv(fpath, sep='\t')

def load_pwms():
    """
    Loads list of motif objects from pickle file, rather than using PyJASPAR
    Return: List of motif objects
    """
    
    fpath = 'data/pwms.pkl'
    with open(fpath, 'rb') as file:
        motifs = pickle.load(file)
    return motifs

def preprocess(motifs, class_threshold=20, print_info=False):
    
    """
    Pre-processing of motif list to remove dimers, and select 'main_classes'
    Return: pwm np matrices, family_id, matrix_id, id_to_fam dict, and n_fams
    """
    
    # Remove dimers
    motifs = [motif for motif in motifs if len(motif.tf_class) == 1]

    # Creating list of main_classes
    all_classes = [motif.tf_class[0] for motif in motifs]
    sorted_classes = Counter(all_classes).most_common()
    main_classes = []
    for classname in sorted_classes:
        value,count = classname
        if count >= class_threshold:
            main_classes.append(value)
    
    data = [] # pwms
    labels = [] # tf fams
    matrix_ids = []

    # Generating datasets
    for motif in motifs:
        tf_class = motif.tf_class[0]
        pwm = np.array(list(motif.pwm.values()))
        if tf_class in main_classes:
            labels.append(tf_class)
            data.append(pwm)
            matrix_ids.append(motif.matrix_id)
             
    # numerical class labels
    class_to_num = {label: index for index, label in enumerate(main_classes)}
    num_labels = [class_to_num[label] for label in labels]
    num_to_class = {v: k for k, v in class_to_num.items()}
    counted_labels = Counter(labels)
    
    # numerical class labels with counts
    num_to_class_cnts = {}
    for (k,v) in num_to_class.items():
        new_v = f'{v} ({counted_labels[v]})'
        num_to_class_cnts[k] = new_v

    if print_info == True:
        print(f'\nCounts: {counted_labels}')
        # print(f'\nNumerical labels: {class_to_num}')
        print(f'\nNumerical f1_scorelabels with counts: {num_to_class_cnts}')
        print(f'\nNumber of PWMs: {len(data)}')
        print(f'\nNumber of fams: {len(main_classes)}')

    
    return data, num_labels, matrix_ids, num_to_class, class_to_num, main_classes

def get_target_dict(m_ids):
    """
    Stores motif object for all tomtom matches.
    """
    jdb_obj = jaspardb()
    target_dict = {}
    for m_id in m_ids:
        if isinstance(m_id, str):
            motif = jdb_obj.fetch_motif_by_id(m_id)
            target_dict[m_id] = motif
    return target_dict

def majority_vote(n_classes):

    if not n_classes:
        return None
    
    counter = Counter(n_classes)
    max_count = max(counter.values())

    modes = [item for item, cnt in counter.items() if cnt == max_count]

    if len(modes) == 1:
        return modes[0]
    else:
        return majority_vote(n_classes[:-1])

def run_knn(k, df, target_dict, main_classes, m_ids, fams, fam_to_id):

    y_pred = []
    fam_to_id['Other'] = 20
    jdb_obj = jaspardb()

    q_ids = df['Query_ID'].unique()
    for i,m_id in enumerate(m_ids):
        if m_id in q_ids:

            # Ranked List of Targets
            df_sub = df[df['Query_ID']==m_id] # Make subset df
            t_ids = df_sub['Target_ID'].tolist() # Get ranked list of targets
            t_ids = [t for t in t_ids if t[:-2] != m_id[:-2]] # Remove self from target_ids

            # K nearest neighbor classes
            n_classes = []
            i = 0
            while len(n_classes) < k and i < len(t_ids):
                if t_ids[i] in target_dict:
                    m = target_dict[t_ids[i]]
                else:
                    m = jdb_obj.fetch_motif_by_id(t_ids[i])
                if m:
                    cls = m.tf_class
                    if len(cls) == 1:
                        if cls[0] in main_classes:
                            n_classes.append(cls[0])
                        else:
                            n_classes.append('Other')
                    else:
                        n_classes.append('Other')
                i += 1
            if not n_classes:
                n_classes.append('Other')

            # Majority vote
            pred = majority_vote(n_classes)
            y_pred.append(fam_to_id[pred])
        
        else:
            y_pred.append(20)

    accuracy = accuracy_score(fams, y_pred)
    f1 = f1_score(fams, y_pred, average='macro')
    return accuracy, f1, y_pred

def total_knn(k, df1, df2, df3, target_dict, main_classes, m_ids, fams, fam_to_id):
    """
    Runs all 3 knn's to make an ensemble, then reports y_pred of all of them. Can spit out any desired y_pred, as well.
    """

    accuracy1, f1_1, y_pred1 = run_knn(k, df1, target_dict, main_classes, m_ids, fams, fam_to_id)
    accuracy2, f1_2, y_pred2 = run_knn(k, df2, target_dict, main_classes, m_ids, fams, fam_to_id)
    accuracy3, f1_3, y_pred3 = run_knn(k, df3, target_dict, main_classes, m_ids, fams, fam_to_id)

    y_pred_ensemble = [majority_vote([a,b,c]) for a,b,c in zip(y_pred1,y_pred2,y_pred3)]

    accuracy = accuracy_score(fams, y_pred_ensemble)
    f1 = f1_score(fams, y_pred_ensemble, average='macro')
    return y_pred1, y_pred2, y_pred3, y_pred_ensemble