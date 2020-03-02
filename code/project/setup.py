from datetime import date
from pathlib import Path

# [Google Drive]
#from google.colab import drive
#drive.mount('/content/drive')
#project_path = '/content/drive/My Drive/hp/translation/baseline/gru_attention/'
colab = False

# [Local]
project_path = "S:/_UNI/HP/MT/TL/notebooks/gru-attention/"
#project_path = "C:/keras/gru_attention/"

model_version  = '02'
current_date   = str(date.today())
main_dir       = project_path + 'models/' + current_date + '/' + model_version + '/'
log_dir        = main_dir + 'logs/'
results_path   = main_dir + 'results/' 
model_dir      = main_dir + 'model/'

model_path      = model_dir + 'model.h5'
encoder_path    = model_dir + 'encoder.json'
encoder_w_path  = model_dir + 'encoder_weights.h5'
decoder_path    = model_dir + 'decoder.json'
decoder_w_path  = model_dir + 'decoder_weights.h5'
model_img_path  = model_dir + 'model_diagram.png'

#source_language_name = "Gaelic"
#target_language_name = "English"
#data_input_path      = project_path + "data/gd-en/"
#data_output_path     = main_dir + "data_output/"
#source_language_txt  = data_input_path + "gd.txt"
#target_language_txt  = data_input_path + "en.txt"

source_language_name = "Irish"
target_language_name = "English"
data_input_path      = project_path + "data/ga-en/"
data_output_path     = main_dir + "data_output/"
source_language_txt  = data_input_path + "ga.txt"
target_language_txt  = data_input_path + "en.txt"

#source_language_name = "French"
#target_language_name = "English"
#data_input_path      = project_path + "data/fr-en/small/"
#data_output_path     = main_dir + "data_output/"
#source_language_txt  = data_input_path + "fr.txt"
#target_language_txt  = data_input_path + "en.txt"

# make the directories if they don't exist
Path(main_dir).mkdir(parents=True, exist_ok=True)
Path(model_dir).mkdir(parents=True, exist_ok=True)
Path(log_dir).mkdir(parents=True, exist_ok=True)
Path(data_output_path).mkdir(parents=True, exist_ok=True)
Path(results_path).mkdir(parents=True, exist_ok=True)

