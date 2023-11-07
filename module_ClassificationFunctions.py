# packages
import numpy as np
import matplotlib.pyplot as plt

def generateGraphReport(classificationReport: dict, numData: int ,save: bool = True):
    metrics = ['precision', 'recall', 'f1-score', 'accuracy']
    classes = ['0.0', '1.0']
    colors = ['#0b5a2b', '#498c71']
    fig, axs = plt.subplots(1,4, figsize=(20,8))
    axs = axs.flatten()

    for idx, m in enumerate(metrics):
        if m != 'accuracy':
            measures = []
            for c in classes:
                measures.append(classificationReport[c][m])
            
            x = np.arange(len(classes))
            b = axs[idx].bar(x, measures, width=0.9, color=colors, label=measures)
            axs[idx].bar_label(b, label_type='center', fontsize=15)
            axs[idx].set_title(m, fontsize=15)
            axs[idx].set_xticks(x, ['Normal', 'Abnormal'])
            axs[idx].set_xlabel('Classes')
        else:
            measures = [classificationReport[m]]
            b = axs[idx].bar([0], measures, width=0.1, color=colors[1], label=measures)
            axs[idx].bar_label(b, label_type='center', fontsize=15)
            axs[idx].set_title(m, fontsize=15)
            axs[idx].set_xticks([], color='white')
            axs[idx].set_xlabel('Overall')

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.suptitle(f'ECG Signals \n n = {numData}', fontsize=20)
    plt.show()
    
    if save:
        plt.savefig('ECG_ClassificationReport')

            
