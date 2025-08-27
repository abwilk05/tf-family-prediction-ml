from scripts.functions import set_seed, run_knn, read_tsv, get_target_dict, load_pwms, preprocess, total_knn
from scripts.plots import accuracy, accuracy_FCN, time_FCN, conf_matrix, performance
import matplotlib.pyplot as plt
import pickle

def main():

    set_seed(42)

    motifs = load_pwms()
    pwms, fams, m_ids, id_to_fam, fam_to_id, main_classes = preprocess(motifs, print_info=False)

    # Call functions from functions.py or plots.py as necessary

if __name__ == '__main__':
    main()