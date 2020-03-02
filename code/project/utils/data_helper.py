from project.utils.directories import Info as info
from project.utils.logger import get_logger
from project.utils.vocab import filter_data
import os
import unicodedata
import re
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from pickle import load, dump

logger = get_logger("train", info.log_dir)

# Converts the unicode file to ascii


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(sentence):
    sentence = unicode_to_ascii(sentence.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)

    sentence = sentence.rstrip().strip()

    return sentence


def clean_data(source_sentences, target_sentences, max_words=1):
    source_output, target_output = [], []
    for i in range(len(source_sentences)):
        clean_source, clean_target = preprocess_sentence(
            source_sentences[i]), preprocess_sentence(target_sentences[i])
        if len(clean_source.split()) <= max_words and len(clean_target.split()) <= max_words:
            source_output.append(clean_source)
            target_output.append(clean_target)

    invalid_samples = len(source_sentences)-len(source_output)
    logger.info(
        'Invalid training samples: {}/{}'.format(invalid_samples, len(source_sentences)))
    return source_output, target_output


def visualise_data(tr_source_text, tr_target_text, ts_source_text, ts_target_text):
    source_l, target_l = [], []

    for sentence in tr_source_text:
        source_l.append(len(sentence.split()))
    for sentence in ts_source_text:
        source_l.append(len(sentence.split()))
    for sentence in tr_target_text:
        target_l.append(len(sentence.split()))
    for sentence in ts_target_text:
        target_l.append(len(sentence.split()))

    length_df = pd.DataFrame(
        {info.source_language_name: source_l, info.target_language_name: target_l})
    length_df.hist(bins=30)

    return length_df


def read_data(filename):
    text = []
    with open(filename, 'r', encoding='utf-8') as f:
        for row in f:
            text.append(row)
    return text


def sents2sequences(tokenizer, sentences, reverse=False, pad_length=None, padding_type='post'):
    encoded_text = tokenizer.texts_to_sequences(sentences)
    preproc_text = pad_sequences(
        encoded_text, padding=padding_type, maxlen=pad_length)
    if reverse:
        preproc_text = np.flip(preproc_text, axis=1)

    return preproc_text


def get_data(train_size, test_split, random_seed=100, max_words=None, min_word_occurrence=None):
    # load data from the input files
    source_text = read_data(info.source_language_txt)
    target_text = read_data(info.target_language_txt)
    source_text, target_text = source_text[:
                                           train_size], target_text[:train_size]
    # clean the data
    source_text, target_text = clean_data(
        source_text, target_text, max_words=max_words)
    # filter the data
    source_text, target_text = filter_data(
        source_text, target_text, min_word_occurrence)
    # add the start of string (sos) + end of string (eos) tags
    target_text = ['sos ' + sent[:-1] + 'eos .' if sent.endswith(
        '.') else 'sos ' + sent + ' eos .' for sent in target_text]

    # shuffle the list of indices and split into train + test
    np.random.seed(random_seed)
    inds = np.arange(len(source_text))
    np.random.shuffle(inds)
    train_inds, test_inds = train_test_split(
        inds, test_size=test_split, random_state=20)

    # training data
    tr_source_text = [source_text[ti] for ti in train_inds]
    tr_target_text = [target_text[ti] for ti in train_inds]
    # test data
    ts_source_text = [source_text[ti] for ti in test_inds]
    ts_target_text = [target_text[ti] for ti in test_inds]

    logger.info('Training data size: {}'.format(len(tr_source_text)))
    logger.info('Test data size: {}'.format(len(ts_source_text)))
    return tr_source_text, tr_target_text, ts_source_text, ts_target_text


def split_train_validation(original_source, original_target, validation_split, random_seed=50):
    np.random.seed(random_seed)
    inds = np.arange(len(original_source))
    np.random.shuffle(inds)
    train_inds, validation_inds = train_test_split(
        inds, test_size=validation_split, random_state=10)
    # training data
    tr_source_text = [original_source[ti] for ti in train_inds]
    tr_target_text = [original_target[ti] for ti in train_inds]
    # validation data
    va_source_text = [original_source[ti] for ti in validation_inds]
    va_target_text = [original_target[ti] for ti in validation_inds]
    return tr_source_text, tr_target_text, va_source_text, va_target_text


def convert_data(source_tokenizer, target_tokenizer, source_text, target_text, source_timesteps, target_timesteps):
    # convert the text into to a sequence of word indices
    source_seq = sents2sequences(source_tokenizer, source_text,
                                 reverse=False, padding_type='pre', pad_length=source_timesteps)
    target_seq = sents2sequences(
        target_tokenizer, target_text, pad_length=target_timesteps)
    logger.info('Vocabulary size ({}): {}'.format(
        info.source_language_name, np.max(source_seq)+1))
    logger.info('Vocabulary size ({}): {}'.format(
        info.target_language_name, np.max(target_seq)+1))
    logger.debug('{} text shape: {}'.format(
        info.source_language_name, source_seq.shape))
    logger.debug('{} text shape: {}'.format(
        info.target_language_name, target_seq.shape))
    return source_seq, target_seq


def to_pairs(list_one, list_two):
    output = []
    for i in range(len(list_one)):
        output.append([list_one[i], list_two[i]])
    return output


# save a list of data to a file
def save_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print("Saved: " + filename)


# load cleaned data
def load_data(filename):
  print("Loaded: " + filename)
  return load(open(filename, 'rb'))
