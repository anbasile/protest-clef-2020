import importlib
from pathlib import Path

import tensorflow as tf

from common import ModelType
from data import load_data

from models import define_callbacks

class Trainer:
    def __init__(
        self,
        model_type: ModelType,
        pretrained_model: str,
        max_length:int,
        dataset: Path,
        crf_decoding:bool): 
        """
            TODO
        """
        self.model_type = model_type

        self.pretrained_model = pretrained_model

        self.crf_decoding = crf_decoding

        self.max_length = max_length

        self.output_dir = f'{model_type}_{pretrained_model}_{max_length}_{crf_decoding}'

        module = importlib.import_module('models', self.model_type.name)

        model = getattr(module, self.model_type.name)

        self.model = model(self.pretrained_model, 2, self.crf_decoding)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=3e-5), 
            loss=tf.keras.losses.SparseCategoricalCrossentropy())

        return None

    def run(self):
        """
            TODO
        """

        tfds = load_data(self.pretrained_model, self.max_length)

        self.model.fit(
            tfds, 
            epochs=1,
            callbacks=define_callbacks(self.output_dir))
        return None
