import sys
sys.path.insert(0, '..')
from project.utils.data_helper import read_data, split_data, save_data_text
from project.utils.directories import Info as info
from project.utils.parameters import params

# load the data from the input file
input_file = read_data("S:/_UNI/HP/MT/repo/code/gru_attention/data/ita-en/source/ita.txt")

# split the data from tab delimited
source, target = split_data(input_file)


##############################################

# only use this for monolingual data
filtered_source = set()
for line in source:
    if line not in filtered_source:
        length = len(line.split())
        # get rid of Mary and Tom
        if "Tom" not in line and "Mary" not in line:
            filtered_source.add(line)

##############################################

#output_path = info.data_output_path
output_path = "S:/_UNI/HP/MT/repo/code/gru_attention/data/ita-en/source/"

# save the data
save_data_text(filtered_source, output_path + "split-source.txt")
save_data_text(target, output_path + "split-target.txt", newline=True)
