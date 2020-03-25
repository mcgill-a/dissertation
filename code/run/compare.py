import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_bleu_scores(experiments):
    labels = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']
    ind = np.arange(len(labels))
    width = 0.25 
    
    idx = 0
    for ex in experiments:
        idx += 1
        plt.bar(ind + (width * idx), ex['results'], width, label=ex['label'], color=ex['color'])

    plt.ylabel('BLEU Score')
    plt.xlabel('BLEU Metric')
    plt.title('Translation BLEU Scores')
    plt.xticks(ind + width*2, labels)
    plt.legend(loc='best')
    plt.show()


baseline = {
    'results': [0.1456005981353993, 0.03952732354389121, 0.019308746830021966, 0.003564009885329991],
    'label': 'Baseline',
    'color': '#C70039'
}

trivial = {
    'results': [0.22, 0.09, 0.05, 0.02],
    'label': 'Trivial',
    'color': '#900C3F'
}

hierarchical = {
    'results': [0.28, 0.15, 0.12, 0.08],
    'label': 'Hierarchical',
    'color': '#581845'
}

experiments = []
experiments.append(baseline)
experiments.append(trivial)
experiments.append(hierarchical)

plot_bleu_scores(experiments)