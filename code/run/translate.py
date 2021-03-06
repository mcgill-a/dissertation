# standard imports
import neptune
import math
import sys
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.utils import plot_model
from nltk.translate.bleu_score import corpus_bleu
from tqdm.auto import tqdm

sys.path.insert(0, '..')

# local imports
from project.utils.data_helper import sents2sequences, get_data, split_train_validation, convert_data, visualise_data, to_pairs, save_data, load_data
from project.utils.vocab import to_vocab
from project.utils.logger import get_logger
from project.utils.directories import Info as info
from project.utils.services import timestamp
from project.layers.attention import AttentionLayer
from project.core.train import train
from project.core.inference import infer_nmt
from project.core.model import define_model, save_models, restore_model, restore_models
from project.utils.parameters import params
from private.tokens import NEPTUNE_API_TOKEN


# neptune configuration
neptune.init('mcgill-a/translation', api_token=NEPTUNE_API_TOKEN)
neptune.create_experiment(name='translate-train',
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
DROPOUT_W = params['DROPOUT_W']
DROPOUT_U = params['DROPOUT_U']
source_timesteps = params['SOURCE_TIMESTEPS']
target_timesteps = params['SOURCE_TIMESTEPS']
TEST_SPLIT = params['TEST_SPLIT']
VALIDATION_SPLIT = params['VALIDATION_SPLIT']
MAX_WORDS_PER_SENTENCE = params['MAX_WORDS_PER_SENTENCE']
MIN_WORD_OCCURRENCE = params['MIN_WORD_OCCURRENCE']
DATA_CLEANED = params['DATA_CLEANED']
OVERRIDE_SAVE = params['OVERRIDE_SAVE']
PARENT_EPOCHS = params['PARENT_EPOCHS']

#############################################################################################################################################

logger = get_logger("train", info.log_dir)

log_output = timestamp() + ' | [Stage] - Processing Data'
print(log_output)
neptune.log_text('Runtime', log_output)

# split the input text files into training + test
tr_source_text, tr_target_text, ts_source_text, ts_target_text = get_data(
    data_size=DATA_SIZE, test_split=TEST_SPLIT, max_words=MAX_WORDS_PER_SENTENCE, min_word_occurrence=MIN_WORD_OCCURRENCE, cleaned=DATA_CLEANED)
# visualise the data
data_vis = visualise_data(tr_source_text, tr_target_text,
                          ts_source_text, ts_target_text)
neptune.log_image('Charts', plt.gcf(), image_name="Data Distribution")
plt.clf()

# define the tokenizers (using training data and validation data)
source_tokenizer = keras.preprocessing.text.Tokenizer(oov_token='UNK')
source_tokenizer.fit_on_texts(tr_source_text)

target_tokenizer = keras.preprocessing.text.Tokenizer(oov_token='UNK')
target_tokenizer.fit_on_texts(tr_target_text)

# set the vocabulary size
source_vsize = max(source_tokenizer.index_word.keys()) + 1
target_vsize = max(target_tokenizer.index_word.keys()) + 1

if params['FORCE_SOURCE_VOCAB_SIZE']:
    if source_vsize < params['FORCE_SOURCE_VOCAB_SIZE']:
        source_vsize = params['FORCE_SOURCE_VOCAB_SIZE']

if params['FORCE_TARGET_VOCAB_SIZE']:
    if target_vsize < params['FORCE_TARGET_VOCAB_SIZE']:
        target_vsize = params['FORCE_TARGET_VOCAB_SIZE']

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
save_data(train_data, info.data_output_path + 'train.pkl')
save_data(test_data,  info.data_output_path + 'test.pkl')

#############################################################################################################################################

# define the models
full_model, encoder_model, decoder_model = define_model(
    hidden_size=hidden_size, dropout_w=DROPOUT_W, dropout_u=DROPOUT_U, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
    source_timesteps=source_timesteps, target_timesteps=target_timesteps,
    source_vsize=source_vsize, target_vsize=target_vsize)

plot_model(full_model, to_file=info.model_img_path, show_shapes=True)

history = {'train_loss': [], 'val_loss': []}

# If training is resuming, load the history and models:
if info.models_exist():
    history = load_data(info.model_history)
    full_model, encoder_model, decoder_model = restore_models()

    log_output = timestamp() + \
        ' | [Stage] - Restoring existing trained models and history'
    print(log_output)
    neptune.log_text('Runtime', log_output)

train(N_EPOCHS, full_model, encoder_model, decoder_model, tr_source_seq,
      tr_target_seq, va_source_seq, va_target_seq, BATCH_SIZE, history, source_vsize, target_vsize, neptune, OVERRIDE_SAVE, PARENT_EPOCHS)

#############################################################################################################################################

epochs = range(1,len(history['train_loss'])+1)
x_ticks = [0,2,4,6,8,10,12,14,16,18,20]
y_ticks = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75]

plt.plot(epochs, history['train_loss'], 'g')
plt.plot(epochs, history['val_loss'], 'b')
plt.title('Training & Validation Loss')
plt.ylabel('Loss')
plt.yticks(y_ticks)
plt.xlabel('Epoch')
plt.xticks(x_ticks)
plt.legend(['Train', 'Validation'], loc='upper right')
neptune.log_image('Charts', plt.gcf(),
                  image_name="Training and Validation Loss")
plt.clf()

#############################################################################################################################################

log_output = timestamp() + ' | [Stage] - End'
print(log_output)
neptune.log_text('Runtime', log_output)

neptune.stop()
