import pickle
from pyjaspar import jaspardb
from collections import Counter
import numpy as np

def save_pwms():
    """
    saves list of motif objects as pickle file
    p1: input True if you want to save all versions
    """
    
    jdb_obj = jaspardb()
    motifs = jdb_obj.fetch_motifs(collection='CORE',all_versions=False)

    # # Saving as pickle file
    fpath = 'data/pwms.pkl'
    with open(fpath, 'wb') as file:
        pickle.dump(motifs, file)

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
        print(f'\nNumerical labels with counts: {num_to_class_cnts}')
        print(f'\nNumber of PWMs: {len(data)}')
        print(f'\nNumber of fams: {len(main_classes)}')

    
    return data, num_labels, matrix_ids, num_to_class, num_to_class_cnts, len(main_classes)