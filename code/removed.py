# Inference after training to check attention and output

""" Index2word """
source_index2word = dict(
    zip(source_tokenizer.word_index.values(), source_tokenizer.word_index.keys()))
target_index2word = dict(
    zip(target_tokenizer.word_index.values(), target_tokenizer.word_index.keys()))

""" Inferring with trained model """
test_source = ts_source_text[0]
test_target_actual = ts_target_text[0]

test_source_seq = sents2sequences(
    source_tokenizer, [test_source], pad_length=source_timesteps)
test_target, attn_weights = infer_nmt(
    encoder_model=encoder_model, decoder_model=decoder_model,
    test_source_seq=test_source_seq, source_vsize=source_vsize, target_vsize=target_vsize)
logger.info('Input ({}): {}'.format(info.source_language_name, test_source))
logger.info('Output ({}): {}'.format(
    info.target_language_name, test_target_actual))
logger.info('Translation ({}): {}'.format(
    info.target_language_name, test_target))

""" Attention plotting """
plot_attention_weights(test_source_seq, attn_weights,
                       source_index2word, target_index2word)


###################################################################################################

""" Index2word for attention plot """
source_index2word = dict(
    zip(source_tokenizer.word_index.values(), source_tokenizer.word_index.keys()))
target_index2word = dict(
    zip(target_tokenizer.word_index.values(), target_tokenizer.word_index.keys()))

###################################################################################################



###################################################################################################



###################################################################################################


###################################################################################################