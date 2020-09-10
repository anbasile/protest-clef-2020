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
            self.encoder = TFBertModel.from_pretrained(pretrained_model, from_pt=True)


        self.num_tags = num_labels

        if self.crf_decoding:
            self.transition_params = self.add_weight(
                "transition_params", [self.num_tags, self.num_tags])

        self.output_layer = tf.keras.layers.Dense(
            self.num_tags, name='output_1')

        return None

    def call(self, features, training=None):
        net, _ = self.encoder(features, training=False)
        print(features['input_ids'].shape)

        logits = self.output_layer(net)

        if not training:
            if self.crf_decoding:
                _, tags_prob = tfa.text.crf_decode(
                    logits,
                    self.transition_params,
                    [features['input_ids'].shape[1]] *
                    features['input_ids'].shape[0],  # e.g. [512]*32
                )
            else:
                tags_prob = tf.nn.softmax(logits)

            return tags_prob
        else:
            return logits

    @tf.function
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute our own loss
            if self.crf_decoding:
                log_likelihood, _ = tfa.text.crf_log_likelihood(
                    y_pred,
                    y['output_1'],
                    [x['input_ids'].shape[1]],  # length e.g. 512
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

    def call(self, inputs):
        _, net = self.encoder(inputs, training=False)
        logits = self.output_layer(net)
        return logits


def define_callbacks(output_dir: str):
    """
        TODO
    """
    Path(f'outputs/{output_dir}').mkdir(exist_ok=True, parents=True)

    checkpointing = tf.keras.callbacks.ModelCheckpoint(
        output_dir,
        save_best_only=True,
        monitor='val_loss',
        mode='auto')

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        verbose=0,
        mode='auto')

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir='./logs/tensorboard-logs/',
        histogram_freq=0,
        update_freq=1000)

    callbacks = [
        checkpointing,
        early_stopping,
        tensorboard]

    return callbacks
