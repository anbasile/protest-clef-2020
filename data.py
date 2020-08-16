import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer

import nlp as nld

def load_data(pretrained_model:str, max_length:int):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)

    def encode(examples):
        encoded = tokenizer(
            examples['text'], 
            truncation=True, 
            max_length=max_length, 
            padding=True,
            return_attention_mask=True,
            return_token_type_ids=True)
        return encoded


    dataset = nld.load_dataset('imdb', split='train[:10%]')

    dataset = dataset.map(encode, batched=True)

    feature_columns= ['input_ids', 'attention_mask', 'token_type_ids']

    features = {x: tf.constant(dataset[x]) for x in feature_columns}

    target = {'output_1':dataset['label']}

    tfds = tf.data.Dataset.from_tensor_slices((features, target))

    return tfds.batch(32)