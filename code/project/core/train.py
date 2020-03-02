from project.core.model import save_models
from project.utils.directories import Info as info
from project.utils.data_helper import save_data
from tensorflow.python.keras.utils import to_categorical
from tqdm.auto import tqdm
import numpy as np


def train(epochs, full_model, encoder, decoder, tr_source_seq, tr_target_seq, va_source_seq, va_target_seq, BATCH_SIZE, history, source_vsize, target_vsize, neptune=None):
    prev_epochs = len(history['train_loss'])
    # save history to neptune
    for i in range(len(history['val_loss'])):

        log_output = "[Epoch {}] train_loss: {} | val_loss: {}".format(
            i + 1, history['train_loss'][i], history['val_loss'][i])
        
        neptune.log_text('Runtime', log_output)

        neptune.log_metric('train_loss', i+1, history['train_loss'][i])
        neptune.log_metric('val_loss', i+1, history['val_loss'][i])

    if epochs > 0:

        # train the model
        print("Training Started (1/{})".format(epochs))
        neptune.log_text('Runtime', '[Stage] - Training')
        best_val_loss = None
        for ep in range(prev_epochs, prev_epochs + epochs):
            train_loss = []
            val_loss = []

            # training set
            description = "Epoch {}: (Training)\t".format(ep+1)

            range_ = range(0, tr_source_seq.shape[0] - BATCH_SIZE, BATCH_SIZE)
            total = len(range_)

            with tqdm(total=total, position=0, leave=True) as pbar:
                for bi in tqdm(range_, position=0, leave=True):
                    # one hot encode the current training batch
                    source_onehot_seq = to_categorical(
                        tr_source_seq[bi:bi + BATCH_SIZE, :], num_classes=source_vsize)
                    target_onehot_seq = to_categorical(
                        tr_target_seq[bi:bi + BATCH_SIZE, :], num_classes=target_vsize)
                    # train the current training batch
                    full_model.train_on_batch(
                        [source_onehot_seq, target_onehot_seq[:, :-1, :]], target_onehot_seq[:, 1:, :])
                    # calculate the current training batch loss
                    t_loss = full_model.evaluate([source_onehot_seq, target_onehot_seq[:, :-1, :]], target_onehot_seq[:, 1:, :],
                                                batch_size=BATCH_SIZE, verbose=0)
                    # store the training loss
                    train_loss.append(t_loss)

            # validation set
            description = "Epoch {}: (Validation)\t".format(ep+1)

            range_ = range(0, va_source_seq.shape[0] - BATCH_SIZE, BATCH_SIZE)
            total = len(range_)

            with tqdm(total=total, position=0, leave=True) as pbar:
                for bi in tqdm(range_, position=0, leave=True):
                    # one hot encode the current validation batch
                    source_onehot_seq = to_categorical(
                        va_source_seq[bi:bi + BATCH_SIZE, :], num_classes=source_vsize)
                    target_onehot_seq = to_categorical(
                        va_target_seq[bi:bi + BATCH_SIZE, :], num_classes=target_vsize)
                    # calculate the current validation batch loss
                    v_loss = full_model.evaluate([source_onehot_seq, target_onehot_seq[:, :-1, :]], target_onehot_seq[:, 1:, :],
                                                batch_size=BATCH_SIZE, verbose=0)
                    # store the validation loss
                    val_loss.append(v_loss)

            # store the mean training and validation loss
            if (ep + 1) % 1 == 0:
                mean_train_loss = np.mean(train_loss)
                mean_val_loss = np.mean(val_loss)

                log_output = "[Epoch {}] train_loss: {} | val_loss: {}".format(
                    ep + 1, mean_train_loss, mean_val_loss)

                print(log_output)
                neptune.log_text('Runtime', log_output)
                history['train_loss'].append(mean_train_loss)
                history['val_loss'].append(mean_val_loss)
                save_data(history, info.model_history)

                neptune.log_metric('train_loss', ep+1, mean_train_loss)
                neptune.log_metric('val_loss', ep+1, mean_val_loss)

            # save the models if the validation loss has improved
            if best_val_loss == None:
                best_val_loss = history['val_loss'][0]
                save_models(full_model, encoder, decoder)
            else:
                if best_val_loss > history['val_loss'][ep]:
                    best_val_loss = history['val_loss'][ep]
                    save_models(full_model, encoder, decoder)
