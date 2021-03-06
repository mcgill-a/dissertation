import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_loss_scores_vocab(experiments):
    epochs = range(1,21)
    x_ticks = [0,2,4,6,8,10,12,14,16,18,20]
    y_ticks = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]

    for ex in experiments:
        plt.plot(epochs[:len(ex['loss'])], ex['loss'], ex['color'])

    plt.title('Baseline Vocabulary Size - Validation Loss')
    plt.ylabel('Loss')
    plt.yticks(y_ticks)
    plt.xlabel('Epoch')
    plt.xticks(x_ticks)
    plt.legend(['4000', '5000', '7000'], loc='upper right')
    plt.show()


def plot_loss_scores_length(experiments):
    epochs = range(1,21)
    x_ticks = [0,2,4,6,8,10,12,14,16,18,20]
    y_ticks = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]

    for ex in experiments:
        plt.plot(epochs[:len(ex['loss'])], ex['loss'], ex['color'])

    plt.title('Baseline Sentence Length - Validation Loss')
    plt.ylabel('Loss')
    plt.yticks(y_ticks)
    plt.xlabel('Epoch')
    plt.xticks(x_ticks)
    plt.legend(['15', '20'], loc='upper right')
    plt.show()


def plot_nist_scores(experiments):
    labels = ['NIST-1', 'NIST-2', 'NIST-3', 'NIST-4', 'NIST-5']
    ind = np.arange(len(labels))
    width = 0.25 
    
    idx = 0
    for ex in experiments:
        idx += 1
        plt.bar(ind + (width * idx), ex['nist'], width, label=ex['label'], color=ex['color'])

    plt.ylabel('NIST Score')
    plt.xlabel('NIST Metric')
    plt.title('Translation NIST Scores')
    plt.xticks(ind + width*2, labels)
    plt.legend(loc='best')
    plt.show()


def plot_bleu_scores(experiments):
    labels = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']
    ind = np.arange(len(labels))
    width = 0.25 
    
    idx = 0
    for ex in experiments:
        idx += 1
        plt.bar(ind + (width * idx), ex['bleu'], width, label=ex['label'], color=ex['color'])

    plt.ylabel('BLEU Score')
    plt.xlabel('BLEU Metric')
    plt.title('Translation BLEU Scores')
    plt.xticks(ind + width*2, labels)
    plt.legend(loc='best')
    plt.show()


# 5k vocab 15 sentence length
baseline = {
    'bleu': [],
    'nist': [],
    'label': 'Baseline',
    'color': '#C70039'
}

trivial = {
    'bleu': [],
    'nist': [],
    'label': 'Trivial',
    'color': '#900C3F'
}

hierarchical = {
    'bleu': [],
    'nist': [],
    'label': 'Hierarchical',
    'color': '#581845'
}

vocab4k = {
    'loss': [],
    'bleu': [],
    'label': '4000',
    'color': '#355C7D',
    
}

vocab5k = {
    'loss': [],
    'bleu': [],
    'label': '5000',
    'color': '#6C5B7B',
}

vocab7k = {
    'loss': [],
    'bleu': [],
    'label': '7000',
    'color': '#C06C84',
}


fifteen5k = {
    'loss': [],
    'bleu': [],
    'nist': [],
    'label': '15',
    'color': '#355C7D'
}

twenty5k = {
    'loss': [],
    'bleu': [],
    'nist': [],
    'label': '20',
    'color': '#6C5B7B',
}

vocab = []
vocab.append(vocab4k)
vocab.append(vocab5k)
vocab.append(vocab7k)

sentence = []
sentence.append(fifteen5k)
sentence.append(twenty5k)

plot_loss_scores_vocab(vocab)
plot_bleu_scores(vocab)
plot_loss_scores_length(sentence)
plot_bleu_scores(sentence)
#######################################################


experiments = []
experiments.append(baseline)
experiments.append(trivial)
experiments.append(hierarchical)

plot_bleu_scores(experiments)
plot_nist_scores(experiments)
