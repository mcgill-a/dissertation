from project.utils.directories import Info as info
from project.utils.parameters import params
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
# only set source limit OR target limit, NOT both
def trim_vocab(vocab, min_occurrence, vocab_limit=None):
    tokens = [k for k, c in vocab.items() if c >= min_occurrence]
    # keep reducing the vocab size until it matches
    if vocab_limit:
        if len(tokens) > vocab_limit:
            threshold = 0
            while len(tokens) != vocab_limit:
                tokens = []
                threshold += 1
                for k, count in vocab.items():
                    if len(tokens) == vocab_limit:
                        break
                    else:
                        if count >= threshold:
                            tokens.append(k)

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
    source = filter_lines(info.source_language_name, source_sentences,
                       min_word_occurrence, params['FORCE_SOURCE_VOCAB_SIZE'])
    target = filter_lines(info.target_language_name, target_sentences,
                          min_word_occurrence, params['FORCE_TARGET_VOCAB_SIZE'])
    return source, target


def filter_lines(name, sentences, min_word_occurrence=None, vocab_limit=None):
    lines = sentences
    # calculate vocabulary
    vocab = to_vocab(sentences)
    print(name + ' Vocabulary: %d' % len(vocab))

    # reduce vocabulary
    vocab = trim_vocab(vocab, min_word_occurrence, vocab_limit)
    print('New ' + name + ' Vocabulary: %d' % len(vocab))
    # mark out of vocabulary words
    lines = update_dataset(lines, vocab)

    # save updated dataset
    filename = info.data_output_path + name + '_filtered.pkl'
    # spot check
    for i in range(3):
        print(lines[i])

    return lines
