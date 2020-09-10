import sys
import numpy as np
import tensorflow as tf
from transformers import BertTokenizerFast
from collections import defaultdict

import nlp as nld


class ProtestaData:
    def __init__(self, data_dir, pretrained_model):

        self.task_name = data_dir.split('/')[-1]  # e.g. data/task2 -> 'task2'

        self.data_dir = data_dir

        self.tokenizer = BertTokenizerFast.from_pretrained(pretrained_model)

        self. feature_columns = ['input_ids',
                                 'attention_mask', 'token_type_ids']

        return None

    def load(self):

        if self.task_name == 'task3':
            return self.load_data_for_sequence_tagger()

    def load_data_for_document_classifier(self):
        return None

    def load_data_for_sentence_classifier(self):
        return None

    def load_data_for_sequence_tagger(self, **kwargs):

        LABEL_TOKEN_PAD = 18

        LABEL_SENTENCE_PAD = 19

        BATCH_SIZE = 8

        def encode_test(examples):
            """
                TODO for Angelo: describe what is happening here!
            """

            input_ids = defaultdict(lambda: [101])
            for sentence_id, tokens in enumerate(examples['token']):
                for token in tokens:
                    head, *tail = self.tokenizer.tokenize(token)
                    input_ids[sentence_id].append(
                        self.tokenizer.convert_tokens_to_ids(head))
                    for split in tail:
                        input_ids[sentence_id].append(
                            self.tokenizer.convert_tokens_to_ids(split))

                input_ids[sentence_id].append(102)

                assert len(input_ids[sentence_id]) <= 512

            return {
                'input_ids': list(input_ids.values()),
                'attention_mask': [[1]*len(v) for _, v in input_ids.items()],
                'token_type_ids': [[0]*len(v) for _, v in input_ids.items()],
            }

        def encode_train_and_dev(examples):
            """
                TODO for Angelo: describe what is happening here!
            """

            input_ids = defaultdict(lambda: [101])
            padded_tags = defaultdict(lambda: [LABEL_SENTENCE_PAD])
            for sentence_id, (tokens, tags) in enumerate(zip(examples['token'], examples['label'])):
                assert len(tokens) == len(tags)
                for token, tag in zip(tokens, tags):
                    head, *tail = self.tokenizer.tokenize(token)
                    input_ids[sentence_id].append(
                        self.tokenizer.convert_tokens_to_ids(head))
                    padded_tags[sentence_id].append(tag)
                    for split in tail:
                        input_ids[sentence_id].append(
                            self.tokenizer.convert_tokens_to_ids(split))
                        padded_tags[sentence_id].append(LABEL_TOKEN_PAD)

                input_ids[sentence_id].append(102)
                padded_tags[sentence_id].append(LABEL_SENTENCE_PAD)

                assert len(input_ids[sentence_id]) == len(
                    padded_tags[sentence_id])
                assert len(input_ids[sentence_id]) <= 512

            assert len(input_ids.keys()) == len(padded_tags.keys())

            return {
                'input_ids': list(input_ids.values()),
                'attention_mask': [[1]*len(v) for _, v in input_ids.items()],
                'token_type_ids': [[0]*len(v) for _, v in input_ids.items()],
                'label': list(padded_tags.values())}

        train, dev, test = nld.load_dataset(
            f'{self.data_dir}/protest.py',
            'task3',
            data_dir=self.data_dir,
            split=['train', 'validation', 'test'])

        train = train.map(encode_train_and_dev, batched=True)

        dev = dev.map(encode_train_and_dev, batched=True)

        test = dev.map(encode_test, batched=True)

        train_features = {x: tf.ragged.constant(
            train[x]).to_tensor(0) for x in self.feature_columns}

        train_target = {'output_1': tf.ragged.constant(
            train['label']).to_tensor(LABEL_SENTENCE_PAD)}

        tfds_train = tf.data.Dataset.from_tensor_slices(
            (train_features, train_target)).batch(BATCH_SIZE, drop_remainder=True).prefetch(2)

        dev_features = {x: tf.ragged.constant(
            dev[x]).to_tensor(0) for x in self.feature_columns}

        dev_target = {'output_1': tf.ragged.constant(
            dev['label']).to_tensor(LABEL_SENTENCE_PAD)}

        tfds_dev = tf.data.Dataset.from_tensor_slices(
            (dev_features, dev_target)).batch(BATCH_SIZE, drop_remainder=True).prefetch(2)

        test_features = {x: tf.ragged.constant(
            test[x]).to_tensor(0) for x in self.feature_columns}

        tfds_test = tf.data.Dataset.from_tensor_slices(
            (test_features)).batch(BATCH_SIZE, drop_remainder=False).prefetch(2).repeat(1)

        return tfds_train, tfds_dev, tfds_test
