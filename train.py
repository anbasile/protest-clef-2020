import importlib
import sys
from itertools import chain
from pathlib import Path

import tensorflow as tf

from common import ModelType
from data import ProtestaData
from models import define_callbacks


class Trainer:
    def __init__(
            self,
            model_type: ModelType,
            pretrained_model: str,
            dataset: Path,
            crf_decoding: bool):
        """
            TODO
        """
        self.model_type = model_type
        self.pretrained_model = pretrained_model
        self.crf_decoding = crf_decoding
        self.data_dir = dataset.as_posix()
        self.output_dir = f'{model_type}_{pretrained_model}_{crf_decoding}'

        module = importlib.import_module('models', self.model_type.name)
        model = getattr(module, self.model_type.name)

        num_tags = 2 if model_type.name == 'classifier' else 20

        if model_type.name != 'classifier':
            self.index2label = {
                1: 'B-etime',
                2: 'B-fname',
                3: 'B-loc',
                4: 'B-organizer',
                5: 'B-participant',
                6: 'B-place',
                7: 'B-target',
                8: 'B-trigger',
                9: 'I-etime',
                10: 'I-fname',
                11: 'I-loc',
                12: 'I-organizer',
                13: 'I-participant',
                14: 'I-place',
                15: 'I-target',
                16: 'I-trigger',
                17: 'O',
                18: 'O',
                19: 'O'}

        self.model = model(
            self.pretrained_model,
            num_tags,
            self.crf_decoding)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=3e-5),
            loss=None  # we'll compute the loss inside the model
        )

        return None

    def run(self):
        """
            TODO
        """

        train, dev, test = ProtestaData(
            self.data_dir, self.pretrained_model).load()

        self.model.fit(
            x=train,
            epochs=1,
            validation_data=dev,
            callbacks=define_callbacks(self.output_dir))

        self.model.evaluate(dev)

        encoded_data, nld_df, original_spans = test

        test_predictions = self.model.predict(encoded_data)['predictions']

        # write test predictions to file
        with open(f'{self.output_dir}/test.predictions.tsv', 'w+') as f:
            for tokens, predictions, span in zip(nld_df, test_predictions, list(original_spans.values())):
                tokenized_sentence = list(
                    chain.from_iterable([x[1] for x in span]))
                useful_tags = predictions[1:len(tokenized_sentence)+1]
                cursor = 0
                for original_token, splitted_tokens in span:
                    output_label = self.index2label[useful_tags[cursor:cursor+len(
                        splitted_tokens)][0]]
                    f.write(f'{original_token}\t{output_label}')
                    cursor += len(splitted_tokens)
            f.write('\n')

        return None
