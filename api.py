import json
import pickle
from collections import defaultdict
from typing import List

import numpy as np
import spacy
import tensorflow as tf
from fastapi import FastAPI
from spacy.lang.en import English
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=False)

app = FastAPI()
nlp = spacy.load("en_core_web_sm")

# model = tf.keras.models.load_model('protest/output/event-trigger/')
model = tf.keras.models.load_model('protest/output/arguments/')

# with open('protest/event-trigger-tag-encoder.pickle', 'rb') as f:
with open('protest/arguments-tag-encoder.pickle', 'rb') as f:
    tag_encoder = pickle.load(f)


def predict(sentence: str, seq_len:int):
    encoded_sentence = tokenizer.batch_encode_plus(
        [sentence], pad_to_max_length=True, max_length=200, return_tensors='tf')

    probabilities = model.predict(encoded_sentence)

    yhat_tags = tf.math.argmax(probabilities[0],1).numpy()

    return [tag_encoder.inverse_transform(yhat_tags).tolist()[:seq_len]]



@app.post("/protest")
def protest(docs: List[str]):

    output = defaultdict(list)

    for docid, doc in enumerate(nlp.pipe(docs)):
        for sentence in doc.sents:
            tokens = tokenizer.tokenize(sentence.text)
            labels = predict(tokens, seq_len=len(tokens))
            output[docid].append((tokens,labels))
            print(labels)

    return output

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
