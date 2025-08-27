import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


def accuracy(save=False):

    pcc = [0.8296261239943209, 0.8263132986275438, 0.8149550402271651, 0.8102224325603408, 0.8017037387600567, 0.795551348793185]
    euclid = [0.8400378608613346, 0.8386180785612872, 0.8310459062943681, 0.8253667770941789, 0.8154283009938476, 0.8050165641268339]
    rmse = [0.836251774727875, 0.8386180785612872, 0.8338854708944629, 0.8258400378608614, 0.8159015617605301, 0.807382867960246]
    ensmbl = [0.8419309039280644, 0.8400378608613346, 0.8334122101277804, 0.8267865593942262, 0.8159015617605301, 0.8116422148603881]

    x = range(1,12,2)
    
    plt.plot(x, [y*100 for y in pcc], color='blue',label='PCC')
    plt.plot(x, [y*100 for y in euclid], color='orange',label='Euclidean Dist.')
    plt.plot(x, [y*100 for y in rmse], color='green',label='2-RMSE')
    plt.plot(x, [y*100 for y in ensmbl], color='red',label='KNN Ensemble')

    plt.xlabel('K')
    plt.ylabel('Accuracy (%)')
    plt.ylim(79,85)
    plt.xticks(range(1,12,2))
    plt.legend()
    if not save:
        plt.show()
    else:
        plt.savefig(f'outputs/accuracy.png')
        plt.close()

def performance(save=False):

    pcc = [0.8296261239943209, 0.8263132986275438, 0.8215806909607194, 0.8149550402271651, 0.8130619971604354, 0.8102224325603408, 0.8069096071935636, 0.8017037387600567, 0.7950780880265026, 0.795551348793185]
    euclid = [0.8400378608613346, 0.8386180785612872, 0.836251774727875, 0.8310459062943681, 0.8282063416942735, 0.8253667770941789, 0.8220539517274018, 0.8154283009938476, 0.8097491717936584, 0.8050165641268339]
    sw = [0.836251774727875, 0.8386180785612872, 0.83719829626124, 0.8338854708944629, 0.8319924278277331, 0.8258400378608614, 0.8234737340274492, 0.8159015617605301, 0.8116422148603881, 0.807382867960246]
    ensmbl = [0.8419309039280644, 0.8400378608613346, 0.8386180785612872, 0.8334122101277804, 0.8305726455276857, 0.8267865593942262, 0.8230004732607666, 0.8159015617605301, 0.8130619971604354, 0.8116422148603881]
    pcc_f = [0.7584345865157859, 0.7598439580610769, 0.7611130265348507, 0.7490257356050586, 0.746067857259001, 0.7451775478989747, 0.7416492182106458, 0.7344102603513774, 0.7252654030770097, 0.720650858197135]
    euclid_f = [0.7634786332363314, 0.7571708627568451, 0.7563362859834095, 0.753421346941402, 0.750883070858564, 0.7454172630182667, 0.7402154415932033, 0.7281332466424162, 0.7223024349499922, 0.715554996148871]
    sw_f = [0.7617997705664651, 0.7590900072441126, 0.7580218448235679, 0.753664447120537, 0.7508535653816932, 0.7462953792656906, 0.7430805757444473, 0.7287027315200296, 0.7226477953599102, 0.7084179152619444]
    ensmbl_f = [0.7654116199496412, 0.7602771466643828, 0.7623671095906969, 0.7592101268136343, 0.7528617775183848, 0.7487621563965923, 0.7459923980962304, 0.7334658561743442, 0.7287033889555946, 0.7193472912164207]

    x = [1,3,4,5,6,7,8,9,10,11]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

    ax1.plot(x, pcc, color='blue',label='PCC')
    ax1.plot(x, euclid, color='orange',label='Euclidean Dist.')
    ax1.plot(x, sw, color='green',label='2-RMSE')
    ax1.plot(x, ensmbl, color='red',label='KNN Ensemble')
    ax1.set_ylim(0.79,0.85)
    ax1.set_xticks(range(1,12,2))
    
    ax2.plot(x, pcc_f, color='blue',label='PCC')
    ax2.plot(x, euclid_f, color='orange',label='Euclidean Dist.')
    ax2.plot(x, sw_f, color='green',label='2-RMSE')
    ax2.plot(x, ensmbl_f, color='red',label='KNN Ensemble')
    ax2.set_ylim(0.7,0.78)
    ax1.set_xticks(range(1,12,2))

    # plt.xlabel('K')
    # plt.ylabel('Accuracy (%)')
    ax1.set_ylim(0.79,0.85)
    ax1.set_xticks(range(1,12,2))
    plt.legend()
    if not save:
        plt.show()
    else:
        plt.savefig(f'outputs/performance.png')
        plt.close()

def accuracy_FCN(save=False):
    
    categories = ['Deep Learning', 'PCC', 'Euclidean Dist.', '2-RMSE', 'KNN Ensemble']
    values = [82.25651392632527, 82.96261239943209, 84.00378608613346, 83.6251774727875, 84.19309039280644]
    colors = ['purple', 'gray', 'gray', 'gray', 'gray']
    errors = [3.7121765900132546,0,0,0,0]

    plt.figure(figsize=(8, 5))
    plt.bar(categories, values, yerr=errors, color=colors, capsize=2)
    plt.ylabel("Accuracy (%)")
    plt.ylim(0,100)
    
    if not save:
        plt.show()
    else:
        plt.savefig(f'outputs/accuracy_FCN.png')
        plt.close()

def time_FCN(save=False):
    
    categories = ['Deep Learning', 'PCC', 'Euclidean Dist.', '2-RMSE', 'KNN Ensemble']
    values = [2.3, 3576, 2096, 1990, 7653]
    colors = ['purple', 'gray', 'gray', 'gray', 'gray']

    plt.figure(figsize=(8, 5))
    plt.bar(categories, values, color=colors)
    plt.yscale('log')
    plt.xlabel("Model")
    plt.ylabel("Time (seconds, log scale)")
    
    if not save:
        plt.show()
    else:
        plt.savefig(f'outputs/time_FCN.png')
        plt.close()

def conf_matrix(y_pred,fams,save=False):

    cm = confusion_matrix(fams, y_pred)
    row_sums = cm.sum(axis=1, keepdims=True)
    epsilon = 1e-10
    cm = cm / (row_sums + epsilon)


    plt.figure(figsize=(12,10))
    sns.heatmap(cm*100, annot=True, fmt='.1f', cmap="Blues", cbar=False)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    yticks = np.array(range(20)) + 0.5
    plt.yticks(yticks, range(20), rotation=0)
    plt.tight_layout()
    if not save:
        plt.show()
    else:
        plt.savefig(f'outputs/conf_matrix.png')
        plt.close()