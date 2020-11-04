import sys
from transformers.optimization_tf import AdamWeightDecay, create_optimizer
from models import define_callbacks, MaskedLoss
from data import ProtestaData
from common import ModelType, EncodingMode
import importlib
import os
from itertools import chain
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(42)


class Trainer:
    def __init__(
            self,
            model_type: ModelType,
            pretrained_model: str,
            dataset: Path,
            crf_decoding: bool,
            encoding: EncodingMode):
        """
            TODO
        """
        self.model_type = model_type
        self.pretrained_model = pretrained_model
        self.crf_decoding = crf_decoding
        self.data_dir = dataset.as_posix()
        self.output_dir = f'outputs/{model_type}_{pretrained_model}_{crf_decoding}/'
        self.encoding_mode = encoding

        module = importlib.import_module('models', self.model_type.name)
        model = getattr(module, self.model_type.name)

        num_tags = 2 if model_type.name == 'classifier' else 19

        if model_type.name != 'classifier':
            self.index2label = {
                0: 'B-etime',
                1: 'B-fname',
                2: 'B-loc',
                3: 'B-organizer',
                4: 'B-participant',
                5: 'B-place',
                6: 'B-target',
                7: 'B-trigger',
                8: 'I-etime',
                9: 'I-fname',
                10: 'I-loc',
                11: 'I-organizer',
                12: 'I-participant',
                13: 'I-place',
                14: 'I-target',
                15: 'I-trigger',
                16: 'O',
                17: 'O',
                18: 'O'}

        self.model = model(
            self.pretrained_model,
            num_tags,
            self.crf_decoding)

        if self.crf_decoding:
            self.loss = None
        else:
            self.loss = MaskedLoss()

        return None

    def run(self):
        """
            TODO
        """

        train, dev, _ = ProtestaData(
            self.data_dir, self.pretrained_model, self.encoding_mode).load()

        optimizer, lr = create_optimizer(
            init_lr=1e-3, num_train_steps=len(train), num_warmup_steps=1, weight_decay_rate=0.01)

        self.model.compile(
            optimizer=Adam(learning_rate=2e-5, clipnorm=1.0),
            # optimizer=optimizer,
            metrics=['acc'],
            loss=self.loss,
        )

        self.model.fit(
            x=train,
            epochs=100,
            validation_data=dev,
            callbacks=define_callbacks(self.output_dir))

        self.model.save_weights(
            self.output_dir+'model.saved_model/', save_format='tf')

        return None
