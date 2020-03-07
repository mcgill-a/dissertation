''' Standard Imports '''
import math
import sys
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import tensorflow.keras as keras


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
from project.utils.vocab import to_vocab, trim_vocab, update_dataset
from project.utils.data_helper import sents2sequences, get_data, split_train_validation, convert_data, visualise_data, to_pairs, save_data, load_data, data_info

#############################################################################################################################################

DATA_SIZE = params['DATA_SIZE']
source_timesteps = params['SOURCE_TIMESTEPS']
target_timesteps = params['SOURCE_TIMESTEPS']
TEST_SPLIT = params['TEST_SPLIT']
VALIDATION_SPLIT = params['VALIDATION_SPLIT']
MAX_WORDS_PER_SENTENCE = params['MAX_WORDS_PER_SENTENCE']
MIN_WORD_OCCURRENCE = params['MIN_WORD_OCCURRENCE']
DATA_CLEANED = params['DATA_CLEANED']

#############################################################################################################################################

log_output = timestamp() + ' | [Stage] - Processing Data'
print(log_output)

# split the input text files into training + test
tr_source_text, tr_target_text, ts_source_text, ts_target_text = get_data(
    train_size=DATA_SIZE, test_split=TEST_SPLIT, max_words=MAX_WORDS_PER_SENTENCE, min_word_occurrence=MIN_WORD_OCCURRENCE, cleaned=DATA_CLEANED)
# visualise the data
data_vis = visualise_data(tr_source_text, tr_target_text,
                          ts_source_text, ts_target_text)
plt.show()

# split training data into training + validation
tr_source_text, tr_target_text, va_source_text, va_target_text = split_train_validation(
    tr_source_text, tr_target_text, VALIDATION_SPLIT)

# define tokenizers
source_tokenizer = keras.preprocessing.text.Tokenizer(oov_token='UNK')
source_tokenizer.fit_on_texts(tr_source_text)

target_tokenizer = keras.preprocessing.text.Tokenizer(oov_token='UNK')
target_tokenizer.fit_on_texts(tr_target_text)

# preprocess the data
tr_source_seq, tr_target_seq = convert_data(
    source_tokenizer, target_tokenizer, tr_source_text, tr_target_text, source_timesteps, target_timesteps)
va_source_seq, va_target_seq = convert_data(
    source_tokenizer, target_tokenizer, va_source_text, va_target_text, source_timesteps, target_timesteps)

# display the data info
data_info(tr_source_seq, tr_target_seq)

# set the vocabulary size
source_vsize = max(source_tokenizer.index_word.keys()) + 1
target_vsize = max(target_tokenizer.index_word.keys()) + 1

# convert words to indices
source_index2word = dict(
    zip(source_tokenizer.word_index.values(), source_tokenizer.word_index.keys()))
target_index2word = dict(
    zip(target_tokenizer.word_index.values(), target_tokenizer.word_index.keys()))

# concatenate data into train and test 
train_data = to_pairs(tr_source_text, tr_target_text)
test_data = to_pairs(ts_source_text, ts_target_text)