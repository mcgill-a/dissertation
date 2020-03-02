import os
import numpy as np
import matplotlib.pyplot as plt
from project.parameters import params
from project.utils.directories import Info as info


def plot_attention_weights(encoder_inputs, attention_weights, source_id2word, target_id2word, filename=None):
  """
  Plots attention weights
  :param encoder_inputs: Sequence of word ids (list/numpy.ndarray)
  :param attention_weights: Sequence of (<word_id_at_decode_step_t>:<attention_weights_at_decode_step_t>)
  :param source_id2word: dict
  :param target_id2word: dict
  :return:
  """

  if len(attention_weights) == 0:
    print('Your attention weights was empty. No attention map saved to the disk. ' +
          '\nPlease check if the decoder produced  a proper translation')
    return

  mats = []
  dec_inputs = []
  for dec_ind, attn in attention_weights:
    mats.append(attn.reshape(-1))
    dec_inputs.append(dec_ind)
  attention_mat = np.transpose(np.array(mats))

  fig, ax = plt.subplots(figsize=(32, 32))
  ax.imshow(attention_mat)

  ax.set_xticks(np.arange(attention_mat.shape[1]))
  ax.set_yticks(np.arange(attention_mat.shape[0]))

  ax.set_xticklabels([target_id2word[inp] if inp != 0 else "<Res>" for inp in dec_inputs])
  ax.set_yticklabels([source_id2word[inp] if inp != 0 else "<Res>" for inp in encoder_inputs.ravel()])

  ax.tick_params(labelsize=32)
  ax.tick_params(axis='x', labelrotation=90)

  if filename is None:
    plt.savefig(os.path.join(info.results_path, 'attention.png'))
  else:
    plt.savefig(os.path.join(info.results_path, '{}'.format(filename)))