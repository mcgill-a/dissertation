from project.layers.attention import AttentionLayer
from project.utils.directories import Info as info
import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model, load_model, model_from_json
from tensorflow.keras.optimizers import Adam


def define_model(hidden_size, dropout_w, dropout_u, batch_size, learning_rate, source_timesteps, source_vsize, target_timesteps, target_vsize):
    # Define an input sequence and process it.
    encoder_inputs = Input(batch_shape=(
        batch_size, source_timesteps, source_vsize), name='encoder_inputs')
    decoder_inputs = Input(batch_shape=(
        batch_size, target_timesteps - 1, target_vsize), name='decoder_inputs')

    # Encoder GRU
    encoder_gru = GRU(hidden_size, dropout=dropout_w, recurrent_dropout=dropout_u, return_sequences=True,
                      return_state=True, name='encoder_gru')
    encoder_out, encoder_state = encoder_gru(encoder_inputs)

    # Set up the decoder GRU, using `encoder_states` as initial state.
    decoder_gru = GRU(hidden_size, dropout=dropout_w, recurrent_dropout=dropout_u, return_sequences=True,
                      return_state=True, name='decoder_gru')
    decoder_out, decoder_state = decoder_gru(
        decoder_inputs, initial_state=encoder_state)

    # Attention layer
    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_out, decoder_out])

    # Concat attention input and decoder GRU output
    decoder_concat_input = Concatenate(
        axis=-1, name='concat_layer')([decoder_out, attn_out])

    # Dense layer
    dense = Dense(target_vsize, activation='softmax', name='softmax_layer')
    dense_time = TimeDistributed(dense, name='time_distributed_layer')
    decoder_pred = dense_time(decoder_concat_input)

    adam = Adam(learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

    # Full model
    full_model = Model(
        inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
    full_model.compile(optimizer=adam, loss='categorical_crossentropy')

    full_model.summary()

    """ Inference model """
    batch_size = 1

    """ Encoder (Inference) model """
    encoder_inf_inputs = Input(batch_shape=(
        batch_size, source_timesteps, source_vsize), name='encoder_inf_inputs')
    encoder_inf_out, encoder_inf_state = encoder_gru(encoder_inf_inputs)
    encoder_model = Model(inputs=encoder_inf_inputs, outputs=[
                          encoder_inf_out, encoder_inf_state])

    """ Decoder (Inference) model """
    decoder_inf_inputs = Input(batch_shape=(
        batch_size, 1, target_vsize), name='decoder_word_inputs')
    encoder_inf_states = Input(batch_shape=(
        batch_size, source_timesteps, hidden_size), name='encoder_inf_states')
    decoder_init_state = Input(batch_shape=(
        batch_size, hidden_size), name='decoder_init')

    decoder_inf_out, decoder_inf_state = decoder_gru(
        decoder_inf_inputs, initial_state=decoder_init_state)
    attn_inf_out, attn_inf_states = attn_layer(
        [encoder_inf_states, decoder_inf_out])
    decoder_inf_concat = Concatenate(
        axis=-1, name='concat')([decoder_inf_out, attn_inf_out])
    decoder_inf_pred = TimeDistributed(dense)(decoder_inf_concat)
    decoder_model = Model(inputs=[encoder_inf_states, decoder_init_state, decoder_inf_inputs],
                          outputs=[decoder_inf_pred, attn_inf_states, decoder_inf_state])

    return full_model, encoder_model, decoder_model


def save_models(full_model, encoder, decoder):
    # Save full_model
    full_model.save(info.model_path)
    print("Saved: " + info.model_path)

    # Save encoder_model
    with open(info.encoder_path, 'w+', encoding='utf8') as f:
        f.write(encoder.to_json())
    encoder.save_weights(info.encoder_w_path)
    print("Saved: " + info.encoder_w_path)

    # Save decoder_model
    with open(info.decoder_path, 'w+', encoding='utf8') as f:
        f.write(decoder.to_json())
    decoder.save_weights(info.decoder_w_path)
    print("Saved: " + info.decoder_w_path)

def restore_model(model_filename, model_weights_filename, custom_objects=None):
    with open(model_filename, 'r', encoding='utf8') as f:
        if custom_objects:
            model = model_from_json(f.read(), custom_objects)
        else:
            model = model_from_json(f.read())
    model.load_weights(model_weights_filename)
    print("Loaded: " + model_filename)
    print("Loaded: " + model_weights_filename)
    return model


def restore_models():
    attention_layer_object = {'AttentionLayer': AttentionLayer}
    full_model = tf.keras.models.load_model(info.model_path, attention_layer_object)
    print("Loaded: " + info.model_path)
    encoder_model = restore_model(info.encoder_path, info.encoder_w_path)
    decoder_model = restore_model(info.decoder_path, info.decoder_w_path, attention_layer_object)
    return full_model, encoder_model, decoder_model
