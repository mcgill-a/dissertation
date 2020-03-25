params = {
    'SOURCE_LANGUAGE_NAME': 'Gaelic',
    'TARGET_LANGUAGE_NAME': 'English',
    'SOURCE_LANGUAGE_TEXT': 'U:/_UNI/HP/MT/repo/code/gru_attention/data/gd-en/gd.txt',
    'TARGET_LANGUAGE_TEXT': 'U:/_UNI/HP/MT/repo/code/gru_attention/data/gd-en/en.txt',
    'VERSION': "1001",
    'DESCRIPTION': "Trivial Transfer Learning - Gaelic Child / French Parent",
    'DATA_SIZE': 145000,
    'N_EPOCHS': 1,
    'BATCH_SIZE': 64,
    'HIDDEN_UNITS': 128,
    'LEARNING_RATE': 0.001,
    'DROPOUT_W': 0,
    'DROPOUT_U': 0,
    'SOURCE_TIMESTEPS': 20,
    'TARGET_TIMESTEPS': 20,
    'TEST_SPLIT': 0.2,
    'VALIDATION_SPLIT': 0.2,
    'MAX_WORDS_PER_SENTENCE': 20,
    'MIN_WORD_OCCURRENCE': 2,
    'PROJECT_PATH': "U:/_UNI/HP/MT/repo/code/gru_attention/",
    'DATA_CLEANED': True,
    'TYPE': 'TRIVIAL',
    'FORCE_SOURCE_VOCAB_SIZE': 4500,
    'FORCE_TARGET_VOCAB_SIZE': 4500,
    'OVERRIDE_SAVE': True,
    'PARENT_EPOCHS': 2,
}