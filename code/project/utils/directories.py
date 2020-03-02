from datetime import date
from pathlib import Path
from project.utils.parameters import params
import os


class Info():
    project_path  = params['PROJECT_PATH']
    model_version = params['VERSION']
    current_date  = str(date.today())
    main_dir      = project_path + 'models/' + current_date + '/' + model_version + '/'
    log_dir       = main_dir + 'logs/'
    results_path  = main_dir + 'results/'
    model_dir     = main_dir + 'model/'

    model_path     = model_dir + 'model.h5'
    encoder_path   = model_dir + 'encoder.json'
    encoder_w_path = model_dir + 'encoder_weights.h5'
    decoder_path   = model_dir + 'decoder.json'
    decoder_w_path = model_dir + 'decoder_weights.h5'
    model_img_path = model_dir + 'model_diagram.png'
    model_history   = model_dir + 'history.pkl'
    
    data_output_path     = main_dir + "data_output/"
    source_language_name = params['SOURCE_LANGUAGE_NAME']
    target_language_name = params['TARGET_LANGUAGE_NAME']
    source_language_txt  = params['SOURCE_LANGUAGE_TEXT']
    target_language_txt  = params['TARGET_LANGUAGE_TEXT']

    # make the directories if they don't exist
    Path(main_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(data_output_path).mkdir(parents=True, exist_ok=True)
    Path(results_path).mkdir(parents=True, exist_ok=True)

    @classmethod
    def models_exist(self):
        if not os.path.isfile(self.model_path): return False
        if not os.path.isfile(self.encoder_path): return False
        if not os.path.isfile(self.encoder_w_path): return False
        if not os.path.isfile(self.decoder_path): return False
        if not os.path.isfile(self.decoder_w_path): return False
        if not os.path.isfile(self.model_history): return False
        return True