import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_loss_scores(experiments):
    epochs = range(1,8)
    #epochs = range(1,len(history['train_loss'])+1)
    x_ticks = [0,2,4,6,8,10,12,14,16,18,20]
    y_ticks = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]

    for ex in experiments:
        plt.plot(epochs, ex['loss'], ex['color'])

    #plt.plot(epochs, history['train_loss'], 'g')
    #plt.plot(epochs, history['val_loss'], 'b')

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
    'results': [0.1456005981353993, 0.03952732354389121, 0.019308746830021966, 0.003564009885329991],
    'label': 'Baseline',
    'color': '#C70039'
}

trivial = {
    'results': [0.18398779982030453, 0.020760240373078703, 0.008255049207959284, 0.0018258043741052],
    'label': 'Trivial',
    'color': '#900C3F'
}

hierarchical = {
    'results': [0.28, 0.15, 0.12, 0.08],
    'label': 'Hierarchical',
    'color': '#581845'
}

#######################################################

vocab4k = {
    'loss': [2.17, 1.8, 1.5, 0.9, 0.75, 0.64, 0.6],
    'color': '#355C7D',
}

vocab5k = {
    'loss': [2.05, 1.7, 1.4, 0.85, 0.6, 0.55, 0.5],
    'color': '#6C5B7B',
}

vocab7k = {
    'loss': [1.7070083038582986,1.3661732525456676,1.0724885641540611,0.7679230648180397,
    0.599556667452359, 0.5046975390357866, 0.4522073805661491],
    'color': '#C06C84',
}


vocab = []
vocab.append(vocab4k)
vocab.append(vocab5k)
vocab.append(vocab7k)

plot_loss_scores(vocab)

#######################################################


experiments = []
experiments.append(baseline)
experiments.append(trivial)
#experiments.append(hierarchical)

#plot_bleu_scores(experiments)