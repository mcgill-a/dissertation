import sys
sys.path.insert(0, '..')
from project.utils.data_helper import read_data, clean_data, save_data_text
from project.utils.directories import Info as info
from project.utils.parameters import params
import numpy as np


in_path = "C:/Users/40276245/hp/repo/code/gru_attention/data/ga-en/"
out_path = in_path + 'output/'
files = []

files.append("en.txt")
files.append("ga.txt")

SIZE = 170000

# load data from the input files
source_text = read_data(in_path + files[0])
target_text = read_data(in_path + files[1])

# clean the data
source_text, target_text = clean_data(source_text, target_text, max_words=params['MAX_WORDS_PER_SENTENCE'], preprocess=True)

s_t = [source_text, target_text]

np.random.seed(10)
inds = np.arange(len(source_text))
np.random.shuffle(inds)

out_source = [s_t[0][i] for i in inds]
out_target = [s_t[1][i] for i in inds]

out_source = out_source[:SIZE]
out_target = out_target[:SIZE]

# save the data
save_data_text(out_source, out_path + files[0], False)
save_data_text(out_target, out_path + files[1], False)