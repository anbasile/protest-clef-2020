from pathlib import Path

import tensorflow as tf
import tensorflow_addons as tfa
from transformers import TFBertModel


class SequenceTagger(tf.keras.Model):

    def __init__(
            self,
            pretrained_model: str,
            num_labels: int,
            crf_decoding: bool):
        super(SequenceTagger, self).__init__()

        self.crf_decoding = crf_decoding

        try:
            self.encoder = TFBertModel.from_pretrained(pretrained_model)
        except OSError:
            self.encoder = TFBertModel.from_pretrained(
                pretrained_model, from_pt=True)

        self.num_tags = num_labels

        if self.crf_decoding:
            self.transition_params = self.add_weight(
                "transition_params", [self.num_tags, self.num_tags])

        self.output_layer = tf.keras.layers.Dense(
            self.num_tags, name='output_1')

        return None

    def call(self, features, training=None):
        net = self.encoder(features, training=training)

        logits = self.output_layer(net[0])

        input_shape_ = tf.shape(features['input_ids'])

        sequence_length = tf.tile(
            [input_shape_[1]],  # length
            [input_shape_[0]])  # batch size

        if not training:
            if self.crf_decoding:
                tags_ids, tags_prob = tfa.text.crf_decode(
                    logits,
                    self.transition_params,
                    sequence_length)  # e.g. [512]*32
            else:
                tags_prob = tf.nn.softmax(logits)
                tags_ids = tf.argmax(tags_prob, axis=2)

            return {'logits': logits, 'predictions': tf.cast(tags_ids, tf.int64)}
        else:
            return logits

    @tf.function
    def train_step(self, data):
        x, y = data

        input_shape_ = tf.shape(x['input_ids'])

        sequence_length = tf.tile(
            [input_shape_[1]],  # length
            [input_shape_[0]])  # batch size

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute our own loss
            if self.crf_decoding:
                log_likelihood, _ = tfa.text.crf_log_likelihood(
                    y_pred,
                    y['output_1'],
                    sequence_length,
                    transition_params=self.transition_params)
                batch_size = tf.shape(log_likelihood)[0]
                loss = tf.reduce_sum(-log_likelihood) / \
                    tf.cast(batch_size, log_likelihood.dtype)
            else:
                loss = self.compiled_loss(y['output_1'], y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        return {'loss': loss}

    @tf.function
    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)

        # Updates the metrics tracking the loss
        input_shape_ = tf.shape(x['input_ids'])

        sequence_length = tf.tile(
            [input_shape_[1]],  # length
            [input_shape_[0]])  # batch size

        if self.crf_decoding:
            log_likelihood, _ = tfa.text.crf_log_likelihood(
                y_pred['logits'],
                y['output_1'],
                sequence_length,
                transition_params=self.transition_params)
            batch_size = tf.shape(log_likelihood)[0]
            val_loss = tf.reduce_sum(-log_likelihood) / \
                tf.cast(batch_size, log_likelihood.dtype)
        else:
            val_loss = self.compiled_loss(y['output_1'], y_pred)
        return {'custom_loss': val_loss}


class SequenceClassifier(tf.keras.Model):

    def __init__(
            self,
            pretrained_model: str,
            num_labels: int,
            crf_decoding: bool):
        super(SequenceClassifier, self).__init__()
        self.encoder = TFBertModel.from_pretrained(pretrained_model)
        self.output_layer = tf.keras.layers.Dense(
            units=num_labels, name='output_1')
        return None

    def call(self, inputs, training=None):
        _, net = self.encoder(inputs, training=training)
        logits = self.output_layer(net)
        return logits

class ResetLossCallback(tf.keras.callbacks.Callback):
    """
        We need to manually reset the metrics, since 
        we are using a custom .fit() method.
    """
    def on_epoch_end(self, epoch, logs=None):
        if self.model.crf_decoding:
            for m in self.model.metrics:
                m.reset_states()


def define_callbacks(output_dir: str):
    """
        TODO
    """
    Path(f'{output_dir}').mkdir(exist_ok=True, parents=True)

    checkpointing = tf.keras.callbacks.ModelCheckpoint(
        output_dir,
        save_best_only=True,
        save_weights_only=True,
        monitor='val_custom_loss',
        mode='min')

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_custom_loss',
        patience=6,
        verbose=0,
        restore_best_weights=True,
        mode='min')

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir='./logs/tensorboard-logs/',
        histogram_freq=0,
        update_freq=1000)

    callbacks = [
        checkpointing,
        early_stopping,
        tensorboard,
        ResetLossCallback()
        ]

    return callbacks
