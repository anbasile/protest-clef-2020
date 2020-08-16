from pathlib import Path
import tensorflow as tf
from transformers import TFAutoModel

class SequenceTagger(tf.keras.Model):

    def __init__(
        self,
        pretrained_model:str,
        num_labels:int,
        crf_decoding:bool):
        super(SequenceTagger, self).__init__()
        self.encoder = TFAutoModel.from_pretrained(pretrained_model)
        self.output_layer = tf.keras.layers.Dense(units=num_labels)
        return None

    def call(self, inputs):
        outputs = self.encoder(inputs)
        logits = self.output_layer(outputs[0])
        return logits

class SequenceClassifier(tf.keras.Model):

    def __init__(
        self,
        pretrained_model:str,
        num_labels:int,
        crf_decoding:bool):
        super(SequenceClassifier, self).__init__()
        self.encoder = TFAutoModel.from_pretrained(pretrained_model)
        self.output_layer = tf.keras.layers.Dense(units=num_labels, activation='softmax', name='output_1')
        return None

    def call(self, inputs):
        outputs = self.encoder(inputs, training=False)
        logits = self.output_layer(outputs[1])
        return logits

def define_callbacks(output_dir:str):
    """
        TODO
    """
    Path(f'ouputs/{output_dir}').mkdir(exist_ok=True, parents=True)

    checkpointing = tf.keras.callbacks.ModelCheckpoint(
        output_dir,
        save_best_only=True,
        monitor='val_loss',
        save_weights_only=True,
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
