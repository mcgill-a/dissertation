import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_loss_scores(experiments):
    epochs = range(1,21)
    #epochs = range(1,len(history['train_loss'])+1)
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


baseline = {
    'bleu': [0.20125178496579257, 0.05802870449811433, 0.03545474454968807, 0.008854346760032756],
    'nist': [0.1695572241336696, 0.01646696917906701, 0.002661757641886449, 0.0000, 0.0000],
    'label': 'Baseline',
    'color': '#C70039'
}

trivial = {
    'bleu': [0.1695572241336696, 0.01646696917906701, 0.002661757641886449, 0.0000],
    'nist': [0.1695572241336696, 0.01646696917906701, 0.002661757641886449, 0.0000, 0.0000],
    'label': 'Trivial',
    'color': '#900C3F'
}

# 5k trivial? 'results': [0.18398779982030453, 0.020760240373078703, 0.008255049207959284, 0.0018258043741052],

hierarchical = {
    'bleu': [0.1695572241336696, 0.01646696917906701, 0.002661757641886449, 0.0000],
    'nist': [0.1695572241336696, 0.01646696917906701, 0.002661757641886449, 0.0000, 0.0000],
    'label': 'Hierarchical',
    'color': '#581845'
}

#######################################################

vocab4k = {
    'loss': [1.7660052358116234, 1.467781496640727, 1.148745621765516,0.9290629078670102,
    0.8275403423203948, 0.7438556449847985, 0.7318846212072267, 0.7111133995635733,
    0.7113203380812597, 0.68858076261552, 0.6705967933762798, 0.6827919760626324,
    0.6505094673587473, 0.6738043677740992, 0.6991398403657734, 0.6832141742027925,
    0.6503916842996745, 0.7088757213308008, 0.7651974666678445, 0.7651974666678445],
    'color': '#355C7D',
}

vocab5k = {
    'loss': [1.6856876430590508,1.3843647287695446,1.1092307116445257,0.8146348814608643,
    0.6044633312778578,0.5052758732717999,0.4453104509668456,0.42456180509862,
    0.4204163138889476,0.4195470244570093,0.4105252346024471,0.4309123474303,
    0.4240458404859401,0.4148839809874141,0.4289870098412312,0.4514234618012,
    0.450994389176493,0.455743739171823,0.4684779376383, 0.487987338108031],
    'color': '#6C5B7B',
}

vocab7k = {
    'loss': [1.7070083038582986,1.3661732525456676,1.0724885641540611,0.7679230648180397,
    0.599556667452359,0.5046975390357866,0.4522073805661491,0.414901299453572,
    0.3953007290870445,0.4229970186619469,0.37754170633646666,0.3771711563619461,
    0.3744923783600001,0.3790544275173825,0.38463621681237087,0.38869207537635253,
    0.39306217164459806,0.4186655496200804,0.4061346339751344],
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
experiments.append(hierarchical)

plot_bleu_scores(experiments)
plot_nist_scores(experiments)