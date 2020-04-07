# standard imports
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, '..')

# local imports
from project.utils.data_helper import save_data, load_data
from project.utils.directories import Info as info

# The purpose of this class it to fix errors in training where models
# aren't saved, meaning the same validation and training loss occurs
# for multiple epochs.

history = {'train_loss': [], 'val_loss': []}

if info.models_exist():
    history = load_data(info.model_history)

def duplicate_check(losses):
    values = []
    duplicate_idx = []
    for i in range(len(losses)):
        if not losses[i] in values:
            values.append(losses[i])
        else:
            duplicate_idx.append(i)

    print(duplicate_idx)

duplicate_check(history['val_loss'])
duplicate_check(history['train_loss'])

print("Duplicates Removed.")
print(history['val_loss'])
duplicate_check(history['val_loss'])
duplicate_check(history['train_loss'])

save_data(history, info.model_history)

epochs = range(1,len(history['train_loss'])+1)
x_epochs = [0,2,4,6,8,10,12,14,16,18,20]
plt.plot(epochs, history['train_loss'], 'g')
plt.plot(epochs, history['val_loss'], 'b')
plt.title('Training & Validation Loss / Epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.xticks(x_epochs)
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
