''' Standard Imports '''
import neptune
import math
import sys
import random
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.utils import plot_model
from nltk.translate.bleu_score import corpus_bleu
from tqdm.auto import tqdm

sys.path.insert(0, '..')

''' Local Imports '''
from private.tokens import NEPTUNE_API_TOKEN
from project.utils.parameters import params
from project.core.model import define_model, save_models, restore_model, restore_models
from project.core.inference import infer_nmt
from project.core.train import train
from project.layers.attention import AttentionLayer
from project.utils.services import timestamp
from project.utils.visualise import plot_attention_weights
from project.utils.directories import Info as info
from project.utils.logger import get_logger
from project.utils.vocab import to_vocab
from project.utils.data_helper import sents2sequences, get_data, split_train_validation, convert_data, visualise_data, to_pairs, save_data, load_data


''' Neptune Configuration '''
neptune.init('mcgill-a/translation', api_token=NEPTUNE_API_TOKEN)
neptune.create_experiment(name='translate-evaluate',
                          params=params)

log_output = timestamp() + ' | [Stage] - Start'
print(log_output)
neptune.log_text('Runtime', log_output)
#############################################################################################################################################

DATA_SIZE = params['DATA_SIZE']
N_EPOCHS = params['N_EPOCHS']
BATCH_SIZE = params['BATCH_SIZE']
hidden_size = params['HIDDEN_UNITS']
LEARNING_RATE = params['LEARNING_RATE']
source_timesteps = params['SOURCE_TIMESTEPS']
target_timesteps = params['SOURCE_TIMESTEPS']
TEST_SPLIT = params['TEST_SPLIT']
VALIDATION_SPLIT = params['VALIDATION_SPLIT']
MAX_WORDS_PER_SENTENCE = params['MAX_WORDS_PER_SENTENCE']
MIN_WORD_OCCURRENCE = params['MIN_WORD_OCCURRENCE']
DATA_CLEANED = params['DATA_CLEANED']

#############################################################################################################################################

logger = get_logger("train", info.log_dir)

log_output = timestamp() + ' | [Stage] - Processing Data'
print(log_output)
neptune.log_text('Runtime', log_output)


# split the input text files into training + test
tr_source_text, tr_target_text, ts_source_text, ts_target_text = get_data(
    train_size=DATA_SIZE, test_split=TEST_SPLIT, max_words=MAX_WORDS_PER_SENTENCE, min_word_occurrence=MIN_WORD_OCCURRENCE, cleaned=DATA_CLEANED)

# define the tokenizers (using training data and validation data)
source_tokenizer = keras.preprocessing.text.Tokenizer(oov_token='UNK')
source_tokenizer.fit_on_texts(tr_source_text)

target_tokenizer = keras.preprocessing.text.Tokenizer(oov_token='UNK')
target_tokenizer.fit_on_texts(tr_target_text)

# set the vocabulary size
source_vsize = max(source_tokenizer.index_word.keys()) + 1
target_vsize = max(target_tokenizer.index_word.keys()) + 1

# split training data into training + validation
tr_source_text, tr_target_text, va_source_text, va_target_text = split_train_validation(
    tr_source_text, tr_target_text, VALIDATION_SPLIT)


# preprocess the data
tr_source_seq, tr_target_seq = convert_data(
    source_tokenizer, target_tokenizer, tr_source_text, tr_target_text, source_timesteps, target_timesteps)
va_source_seq, va_target_seq = convert_data(
    source_tokenizer, target_tokenizer, va_source_text, va_target_text, source_timesteps, target_timesteps)


# convert the words to indices
source_index2word = dict(
    zip(source_tokenizer.word_index.values(), source_tokenizer.word_index.keys()))
target_index2word = dict(
    zip(target_tokenizer.word_index.values(), target_tokenizer.word_index.keys()))

train_data = to_pairs(tr_source_text, tr_target_text)
test_data = to_pairs(ts_source_text, ts_target_text)

#############################################################################################################################################

""" Defining the full model """
full_model, encoder_model, decoder_model = define_model(
    hidden_size=hidden_size, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
    source_timesteps=source_timesteps, target_timesteps=target_timesteps,
    source_vsize=source_vsize, target_vsize=target_vsize)

plot_model(full_model, to_file=info.model_img_path, show_shapes=True)

history = {'train_loss': [], 'val_loss': []}

# If training is resuming, load the history and models:
if info.models_exist():
    history = load_data(info.model_history)
    full_model, encoder_model, decoder_model = restore_models()
    log_output = timestamp() + ' | [Stage] - Restoring existing trained models and history'
    print(log_output)
    neptune.log_text('Runtime', log_output)
else:
    logger.err("Failed to load model (not found)")

# Train for 0 epochs (just using this to log the restored history)
train(0, full_model, encoder_model, decoder_model, tr_source_seq,
      tr_target_seq, va_source_seq, va_target_seq, BATCH_SIZE, history, source_vsize, target_vsize, neptune)

#############################################################################################################################################


def translate(test_source):
    # inference using the trained encoder and decoder models
    test_source_seq = sents2sequences(
        source_tokenizer, [test_source], pad_length=source_timesteps)
    test_target, attn_weights = infer_nmt(
        encoder_model=encoder_model, decoder_model=decoder_model,
        test_source_seq=test_source_seq, source_vsize=source_vsize,
        target_vsize=target_vsize, target_tokenizer=target_tokenizer, target_index2word=target_index2word)
    return test_target, attn_weights


# evaluate the model
def evaluate_model(test_samples):
    actual, predicted = list(), list()
    iterations = len(test_samples)
    for i in tqdm(range(iterations)):
        raw_src, raw_target = test_samples[i]
        translation, _ = translate(raw_src)
        actual.append([raw_target.split()])
        predicted.append(translation.split())
    # calculate the BLEU score
    bleu_scores = {
        1: corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)),
        2: corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)),
        3: corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)),
        4: corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)),
    }

    for key in bleu_scores:
        neptune.log_metric('BLEU Score', key, bleu_scores[key])
        print('BLEU-', key, ': ', bleu_scores[key])

#############################################################################################################################################

log_output = timestamp() + ' | [Stage] - Evaluating'
print(log_output)
neptune.log_text('Runtime', log_output)
evaluate_model(test_data)

#############################################################################################################################################


def test(test_source, test_target_actual, index=0):
    source_index2word = dict(
        zip(source_tokenizer.word_index.values(), source_tokenizer.word_index.keys()))
    target_index2word = dict(
        zip(target_tokenizer.word_index.values(), target_tokenizer.word_index.keys()))

    # inference using the trained encoder and decoder models
    test_source_seq = sents2sequences(
        source_tokenizer, [test_source], pad_length=source_timesteps)
    test_target, attn_weights = infer_nmt(
        encoder_model=encoder_model, decoder_model=decoder_model,
        test_source_seq=test_source_seq, source_vsize=source_vsize,
        target_vsize=target_vsize, target_tokenizer=target_tokenizer, target_index2word=target_index2word)
    logger.info('Input ({}): {}'.format(info.source_language_name, test_source))
    logger.info('Output ({}): {}'.format(
        info.target_language_name, test_target_actual))
    logger.info('Translation ({}): {}'.format(
        info.target_language_name, test_target))

    # plot the attention
    filename = "attention-" + str(index) + ".png"
    attention_img = plot_attention_weights(test_source_seq, attn_weights,
                        source_index2word, target_index2word, filename=filename)
    neptune.log_image('Attention Plots', attention_img,
                    image_name="Attention Plot " + str(index) )


# Create 5 attention plots
for i in range(5):
    random.seed(i) # get the same 5 random sentences every time
    # idx = random.randrange(len(ts_source_text))
    
    # Only choose from sentences within the first 100 lines.
    # If data size was higher then it might choose
    # an index too high for a smaller subset
    idx = random.randrange(100)
    
    test(ts_source_text[idx], ts_target_text[idx], i+1)

log_output = timestamp() + ' | [Stage] - End'
print(log_output)
neptune.log_text('Runtime', log_output)
neptune.stop()
