import sys
sys.path.insert(0, '..')
from project.utils.data_helper import read_data, split_data, save_data_text
from project.utils.directories import Info as info
from project.utils.parameters import params

# load the data from the input file
input_file = read_data("C:/Users/40276245/hp/repo/code/gru_attention/data/ga-en/tab/en-ga.txt")

# split the data from tab delimited
source, target = split_data(input_file)

# save the data
save_data_text(source, info.data_output_path + "split-source.txt")
save_data_text(target, info.data_output_path + "split-target.txt", newline=True)


