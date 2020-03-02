from project.utils.data_helper import sents2sequences
from tensorflow.python.keras.utils import to_categorical
import numpy as np


def infer_nmt(encoder_model, decoder_model, test_source_seq, source_vsize, target_vsize, target_tokenizer, target_index2word):
    """
    Infer logic
    :param encoder_model: keras.Model
    :param decoder_model: keras.Model
    :param test_source_seq: sequence of word ids
    :param source_vsize: int
    :param target_vsize: int
    :return:
    """

    test_target_seq = sents2sequences(target_tokenizer, ['sos'], target_vsize)
    test_source_onehot_seq = to_categorical(
        test_source_seq, num_classes=source_vsize)
    test_target_onehot_seq = np.expand_dims(
        to_categorical(test_target_seq, num_classes=target_vsize), 1)

    enc_outs, enc_last_state = encoder_model.predict(test_source_onehot_seq)
    dec_state = enc_last_state
    attention_weights = []
    target_text = ''
    for i in range(20):
        dec_out, attention, dec_state = decoder_model.predict(
            [enc_outs, dec_state, test_target_onehot_seq])
        dec_ind = np.argmax(dec_out, axis=-1)[0, 0]

        if dec_ind == 0:
            break
        test_target_seq = sents2sequences(
            target_tokenizer, [target_index2word[dec_ind]], target_vsize)
        test_target_onehot_seq = np.expand_dims(
            to_categorical(test_target_seq, num_classes=target_vsize), 1)

        attention_weights.append((dec_ind, attention))
        target_text += target_index2word[dec_ind] + ' '

    return target_text, attention_weights
