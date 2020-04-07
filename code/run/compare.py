import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_scores(experiments):
    epochs = range(1,8)
    x_ticks = [0,2,4,6,8,10,12,14,16,18,20]
    y_ticks = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]

    for ex in experiments:
        plt.plot(epochs, ex['loss'], ex['color'])

    plt.title('Baseline Vocabulary Size - Validation Loss')
    plt.ylabel('Loss')
    plt.yticks(y_ticks)
    plt.xlabel('Epoch')
    plt.xticks(x_ticks)
    plt.legend(['4000', '5000', '7000'], loc='upper right')
    plt.show()


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
    'results': [],
    'label': 'Baseline',
    'color': '#C70039'
}

trivial = {
    'results': [],
    'label': 'Trivial',
    'color': '#900C3F'
}

hierarchical = {
    'results': [],
    'label': 'Hierarchical',
    'color': '#581845'
}

vocab4k = {
    'loss': [],
    'color': '#355C7D',
}

vocab5k = {
    'loss': [],
    'color': '#6C5B7B',
}

vocab7k = {
    'loss': [],
    'color': '#C06C84',
}

vocab = []
vocab.append(vocab4k)
vocab.append(vocab5k)
vocab.append(vocab7k)

plot_loss_scores(vocab)

experiments = []
experiments.append(baseline)
experiments.append(trivial)
experiments.append(hierarchical)

plot_bleu_scores(experiments)