from project.utils.directories import Info as info
from pickle import load, dump
from collections import Counter


# create a frequency table for all words
def to_vocab(lines):
    vocab = Counter()
    for line in lines:
        tokens = line.split()
        vocab.update(tokens)
    return vocab


# remove all words with a frequency below a threshold
def trim_vocab(vocab, min_occurrence):
    tokens = [k for k, c in vocab.items() if c >= min_occurrence]
    return set(tokens)


# mark all OOV with "unk" for all lines
def update_dataset(lines, vocab):
    new_lines = list()
    for line in lines:
        new_tokens = list()
        for token in line.split():
            if token in vocab:
                new_tokens.append(token)
            else:
                new_tokens.append('unk')
        new_line = ' '.join(new_tokens)
        new_lines.append(new_line)
    return new_lines


def filter_data(source_sentences, target_sentences, min_word_occurrence=None):
    return filter_lines(info.source_language_name, source_sentences, min_word_occurrence), filter_lines(info.target_language_name, target_sentences, min_word_occurrence)


def filter_lines(name, sentences, min_word_occurrence=None):
    lines = sentences
    # calculate vocabulary
    vocab = to_vocab(sentences)
    print(name + ' Vocabulary: %d' % len(vocab))
    # reduce vocabulary
    vocab = trim_vocab(vocab, min_word_occurrence)
    print('New ' + name + ' Vocabulary: %d' % len(vocab))
    # mark out of vocabulary words
    lines = update_dataset(lines, vocab)

    # save updated dataset
    filename = info.data_output_path + name + '_filtered.pkl'
    # spot check
    for i in range(3):
        print(lines[i])

    return lines
