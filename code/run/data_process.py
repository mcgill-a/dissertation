import sys
sys.path.insert(0, '..')
from project.utils.data_helper import read_data, clean_data, save_data
from project.utils.directories import Info as info
from project.utils.parameters import params

# load data from the input files
source_text = read_data(info.source_language_txt)
target_text = read_data(info.target_language_txt)

# clean the data
source_text, target_text = clean_data(source_text, target_text, max_words=params['MAX_WORDS_PER_SENTENCE'])

# save the data
save_data(source_text, info.data_output_path + "source.txt")
save_data(target_text, info.data_output_path + "target.txt")