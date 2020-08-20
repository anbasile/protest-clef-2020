import importlib
from pathlib import Path

import tensorflow as tf
from transformers import AdamWeightDecay

from common import ModelType
from data import load_data
from models import define_callbacks


class Trainer:
    def __init__(
            self,
            model_type: ModelType,
            pretrained_model: str,
            max_length: int,
            dataset: Path,
            crf_decoding: bool):
        """
            TODO
        """
        self.model_type = model_type
        self.pretrained_model = pretrained_model
        self.crf_decoding = crf_decoding
        self.max_length = max_length
        self.data_dir = dataset.as_posix()
        self.output_dir = f'{model_type}_{pretrained_model}_{max_length}_{crf_decoding}'
        module = importlib.import_module('models', self.model_type.name)
        model = getattr(module, self.model_type.name)

        num_tags = 2 if model_type.name == 'classifier' else 17

        self.model = model(
            self.pretrained_model,
            num_tags,
            self.crf_decoding)

        if crf_decoding:
            loss = None  # we'll write the loss inside the model
        else:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=3e-5),
            loss=loss)

        self.model.summary()
        return None

    def run(self):
        """
            TODO
        """

        tfds = load_data(self.data_dir, self.pretrained_model, self.max_length)

        self.model.fit(
            x=tfds,
            epochs=2,
            callbacks=define_callbacks(self.output_dir))

        self.model.predict(tfds)
        return None
