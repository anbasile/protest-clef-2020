import importlib
import math
import os
import sys
from collections import defaultdict
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizerFast

from common import ModelType
from data import ProtestaData
from models import SequenceClassifier, SequenceTagger

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

class Inferencer:
    def __init__(
            self,
            model_dir: Path,
            input_file: Path):
        """
            TODO
        """

        self.model_type, self.pretrained_model, self.crf_decoding = model_dir.name.split('_')


        self.input_file = input_file

        self.output_file_name = input_file.with_suffix(f'.{self.model_type}-{self.pretrained_model}-{self.crf_decoding}')

        num_tags = 2 if self.model_type == 'classifier' else 19

        if self.model_type != 'classifier':
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

        self.tokenizer = BertTokenizerFast.from_pretrained(self.pretrained_model)

        if self.model_type == 'tagger':
            self.model = SequenceTagger(self.pretrained_model, num_tags, self.crf_decoding)
        elif self.model_type == 'classifier':
            self.model = SequenceClassifier(self.pretrained_model, num_tags)

        self.model.load_weights(f'{model_dir.as_posix()}/model.saved_model/')

        print(f'Running inference on {input_file} using {model_dir.name}')
        self.load_tokenized_data()

    def load_tokenized_data(self):
        df = pd.read_table(self.input_file, quoting=3, names=['token'])

        df['splits'] = df.token.apply(self.tokenizer.tokenize)

        df['ids'] = df.splits.apply(self.tokenizer.convert_tokens_to_ids)

        df['sentence_id'] = df.token.str.contains('SAMPLE_START').astype(int).cumsum()-1

        df = df[~df.token.isin(['SAMPLE_START', '[SEP]'])]

        sentence_grouped = df.groupby('sentence_id')

        self.df = list(chain.from_iterable(np.array_split(g, math.ceil(g.ids.apply(len).sum()/510)) for _, g in sentence_grouped))

        input_ids = [np.concatenate([
            np.array([101]),
            chunk.explode('ids').ids.values,
            np.array([102])]) for chunk in self.df]

        encoded_data = tf.data.Dataset.from_tensor_slices({
            'input_ids' : tf.ragged.constant(input_ids).to_tensor(0),
            'attention_mask' : tf.ragged.constant([[1]*len(x) for x in input_ids]).to_tensor(0),
            'token_type_ids' : tf.ragged.constant([[0]*len(x) for x in input_ids]).to_tensor(0),
        })

        return encoded_data.batch(8)

    def run(self):

        data = self.load_tokenized_data()

        predictions = self.model.predict(data)['predictions']

        output_lines = []
        for chunk_id, chunk in enumerate(self.df):
            tmp = chunk.explode('ids')
            tmp['predictions'] = predictions[chunk_id][1:tmp.shape[0]+1]
            for n, g in tmp.groupby(tmp.index):
                output_lines.append(f'{g.token.iloc[0]}\t{self.index2label[g.predictions.iloc[0]]}')

        with open(self.input_file, 'r') as f:
            for idx, line in enumerate(f):
                if line.strip() in ['SAMPLE_START', '[SEP]', '']:
                    output_lines.insert(idx, line.strip())

        with open(self.output_file_name, 'w') as f:
            for line in output_lines:
                f.write(f'{line}\n')

