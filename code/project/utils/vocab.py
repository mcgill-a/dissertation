from project.utils.directories import Info as info
from project.utils.parameters import params
from pickle import load, dump
from collections import Counter
import collections


# create a frequency table for all words
def to_vocab(lines):
    vocab = Counter()
    for line in lines:
        tokens = line.split()
        vocab.update(tokens)
    return vocab


# remove all words with a frequency below a threshold
def trim_vocab(vocab, min_occurrence, vocab_limit=None):
    if min_occurrence == None:
        min_occurrence = 1
    sorted_vocab = vocab.items()
    if vocab_limit:
        # sort the vocab from most frequent to least frequent
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    
    tokens = [k for k, c in sorted_vocab if c >= min_occurrence]
    
    # enforce the vocab size limit
    if vocab_limit and len(tokens) > vocab_limit:
        limited_tokens = []
        for word in tokens:
            if len(limited_tokens) == vocab_limit+1:
                break
            else:
                limited_tokens.append(word)
        return limited_tokens
    return tokens


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


def filter_data(source_sentences, target_sentences, min_word_occurrence=None, limit=None):

    if limit:
        source = filter_lines(info.source_language_name, source_sentences,
                              min_word_occurrence, params['FORCE_SOURCE_VOCAB_SIZE'])
        target = filter_lines(info.target_language_name, target_sentences,
                              min_word_occurrence, params['FORCE_TARGET_VOCAB_SIZE'])
    else:
        source = filter_lines(info.source_language_name, source_sentences,
                              min_word_occurrence)
        target = filter_lines(info.target_language_name, target_sentences,
                              min_word_occurrence)
    return source, target


def filter_lines(name, sentences, min_word_occurrence=None, vocab_limit=None):
    lines = sentences
    # calculate vocabulary
    vocab = to_vocab(sentences)
    print(name + ' Vocabulary: %d' % len(vocab))

    # reduce vocabulary
    vocab = trim_vocab(vocab, min_word_occurrence, vocab_limit)
    lines = update_dataset(lines, vocab)
    if vocab_limit:
        print('New ' + name + ' Vocabulary: %d' % len(vocab))
    # mark out of vocabulary words
    

    # spot check
    for i in range(3):
        print(lines[i])

    return lines
