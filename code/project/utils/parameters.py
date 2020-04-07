params = {
    'SOURCE_LANGUAGE_NAME': 'Gaelic',
    'TARGET_LANGUAGE_NAME': 'English',
<<<<<<< HEAD
    'SOURCE_LANGUAGE_TEXT': 'C:/Users/40276245/hp/repo/code/gru_attention/data/gd-en/gd.txt',
    'TARGET_LANGUAGE_TEXT': 'C:/Users/40276245/hp/repo/code/gru_attention/data/gd-en/en.txt',
    'VERSION': "3002",
    'DESCRIPTION': "Hierarchical Transfer - French > Irish > Gaelic (No Dropout)",
    'DATA_SIZE': 145000,
=======
    'SOURCE_LANGUAGE_TEXT': 'U:/_UNI/HP/MT/repo/code/gru_attention/data/gd-en/gd.txt',
    'TARGET_LANGUAGE_TEXT': 'U:/_UNI/HP/MT/repo/code/gru_attention/data/gd-en/en.txt',
    'VERSION': "5000",
    'DESCRIPTION': "Baseline Gaelic",
    'DATA_SIZE': 139000,
>>>>>>> 914c3c338baf4609d68773ea6f6c19d5032fc057
    'N_EPOCHS': 1,
    'BATCH_SIZE': 64,
    'HIDDEN_UNITS': 128,
    'LEARNING_RATE': 0.001,
    'DROPOUT_W': 0,
    'DROPOUT_U': 0,
<<<<<<< HEAD
    'SOURCE_TIMESTEPS': 20,
    'TARGET_TIMESTEPS': 20,
=======
    'SOURCE_TIMESTEPS': 15,
    'TARGET_TIMESTEPS': 15,
>>>>>>> 914c3c338baf4609d68773ea6f6c19d5032fc057
    'TEST_SPLIT': 0.2,
    'VALIDATION_SPLIT': 0.2,
    'MAX_WORDS_PER_SENTENCE': 15,
    'MIN_WORD_OCCURRENCE': 8,
    'PROJECT_PATH': "U:/_UNI/HP/MT/repo/code/gru_attention/",
    'DATA_CLEANED': True,
<<<<<<< HEAD
    'TYPE': 'CHILD',
    'FORCE_SOURCE_VOCAB_SIZE': 7000,
    'FORCE_TARGET_VOCAB_SIZE': 7000,
    'OVERRIDE_SAVE': True,
    'PARENT_EPOCHS': 10,
=======
    'TYPE': 'BASELINE',
    'FORCE_SOURCE_VOCAB_SIZE': None,
    'FORCE_TARGET_VOCAB_SIZE': None,
    'OVERRIDE_SAVE': True,
    'PARENT_EPOCHS': 0,
>>>>>>> 914c3c338baf4609d68773ea6f6c19d5032fc057
}