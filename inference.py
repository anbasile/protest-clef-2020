import importlib
import os
import sys
from collections import defaultdict
from itertools import chain
from pathlib import Path

import numpy as np
import tensorflow as tf

from common import ModelType
from data import ProtestaData
from models import define_callbacks
from transformers import BertTokenizerFast

os.environ['CUDA_VISIBLE_DEVICES'] = "2"


class Inferencer:
    def __init__(
            self,
            model_dir: Path,
            input_file: Path):
        """
            TODO
        """
        config_ = model_dir.name.split('_')
        self.MAX_LENGTH = 512
        self.model_type = config_[0]
        self.pretrained_model = config_[1]
        self.crf_decoding = config_[2]
        self.input_file = input_file
        self.output_file = f'{input_file.name}_from_{model_dir.name}.predictions.tsv'

        self.model = tf.saved_model.load(model_dir.as_posix())
        self.tokenizer = BertTokenizerFast.from_pretrained(self.pretrained_model)

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

        return None

    def run(self):
        """
            TODO
        """
        with open(self.input_file, 'r') as f:
            data = defaultdict(list)
            sentence_id = 0
            for line in f:
                token = line.strip()
                if token == '':
                    sentence_id +=1
                else:
                    data[sentence_id].append({token: self.tokenizer.tokenize(token)})

            with open(self.output_file, 'w+') as f:
                for sentence_id, instance in data.items():
                    print(f'Sentence: {sentence_id}')
                    tmp = np.array(list(chain.from_iterable(list(chain.from_iterable((x.values() for x in instance))))))
                    n_chunks = int(tmp.shape[0] // 483 )+1
                    chunks = np.array_split(tmp, n_chunks)
                    print(len(chunks))
                    predicted_tags = []
                    for chunk in chunks:
                        encoded_tmp = self.tokenizer(chunk.tolist(), is_pretokenized = True, return_tensors='tf', max_length=self.MAX_LENGTH, padding='max_length')
                        tmp_predictions = self.model(encoded_tmp)['predictions'].numpy()[0]
                        tmp_predicted_tags= [self.index2label[i] for i in tmp_predictions[:len(chunk)]]
                        predicted_tags.extend(tmp_predicted_tags)
                    assert len(predicted_tags) == len(tmp)
                    cursor = 0
                    for words in instance:
                        for token, splits in words.items():
                            f.write(f'{token}\t{predicted_tags[cursor:cursor+len(splits)][0]}\n')
                        cursor += len(splits)
                    f.write('\n')

        return None
