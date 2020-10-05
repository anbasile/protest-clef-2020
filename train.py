import importlib
import sys
from itertools import chain
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
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
        self.output_dir = f'outputs/{model_type}_{pretrained_model}_{crf_decoding}/'

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
            loss = None
        else:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=3e-5),
            metrics=['acc'],
            loss=loss,
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
            epochs=100,
            validation_data=dev,
            callbacks=define_callbacks(self.output_dir))

        self.model.save_weights(self.output_dir+'model.saved_model/', save_format='tf')

        # encoded_data, nld_df, original_spans = test

        # test_predictions = self.model.predict(encoded_data)['predictions']

        # # write test predictions to file
        # with open(f'{self.output_dir}/task3_test.predictions.tsv', 'w+') as f:
        #     for tokens, predictions, span in zip(nld_df, test_predictions, list(original_spans.values())):
        #         tokenized_sentence = list(
        #             chain.from_iterable([x[1] for x in span]))
        #         useful_tags = predictions[1:len(tokenized_sentence)+1]
        #         cursor = 0
        #         for original_token, splitted_tokens in span:
        #             output_label = self.index2label[useful_tags[cursor:cursor+len(splitted_tokens)][0]]
        #             f.write(f'{original_token}\t{output_label}\n')
        #             cursor += len(splitted_tokens)
        #         f.write('\n')
        #     f.write('\n')

        return None
