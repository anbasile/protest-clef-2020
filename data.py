import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer

import nlp as nld


def load_data(data_dir: str, pretrained_model: str, max_length: int):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)

    task_name = data_dir.split('/')[-1]  # e.g. data/task2 -> 'task2'

    def encode(examples):

        text_col = {'task1': 'text', 'task2': 'sentence', 'task3': 'token'}

        encoded = tokenizer(
            examples[text_col[task_name]],
            truncation=True,
            max_length=max_length,
            padding=True,
            is_pretokenized=True if task_name == 'task3' else False,
            return_attention_mask=True,
            return_token_type_ids=True)
        return encoded

    # dataset = nld.load_dataset(', split='train[:10%]')
    dataset = nld.load_dataset(
        'data_loader/protest.py',
        task_name,
        data_dir=data_dir)

    dataset = dataset['train']

    dataset = dataset.map(encode, batched=True)

    feature_columns = ['input_ids', 'attention_mask', 'token_type_ids']

    features = {x: tf.ragged.constant(
        dataset[x]).to_tensor(0) for x in feature_columns}

    try:
        target = {'output_1': tf.ragged.constant(
            dataset['label']).to_tensor(0)}
    except AttributeError:
        target = {'output_1': tf.constant(dataset['label'])}

    tfds = tf.data.Dataset.from_tensor_slices((features, target))

    return tfds.batch(32, drop_remainder=True)
