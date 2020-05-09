"""
notes:
- text annotation instead of tables

nice to have:
- fix wrong tags

task 2 train:
- sentence 1, [['O','I','O']] > yes

- sentence 1, [['O','I','O']], no
- sentence 2, [['O','I','O', 'I']], yes

[['B-etime','B-org', 'B-trigger']]

TODO angelo:
- get test data china for task 1 and 2
- predict test data and send to Ali
- evaluate with official script
- collapse subtask 1 and subtask 2 for task 3
- self-supervision experiment

DONE angelo:
- wire model to demo
"""
import json
import re

import requests
import streamlit as st
from newspaper import Article, fulltext
from spacy import displacy

st.title('Protest event detector')

st.markdown('TODO add description\
    ideas: logos, link to paper, abstract?, how to use it?, link to github?')

url = st.text_input('Paste here a sentence or the url of a news article')

def process(text:str):
    r = requests.post(
        url='http://localhost:8000/protest',
        json=[text])
    return r.json()

def align(sentence:list):
    text = []
    ents = []
    tokens = sentence[0] 
    tags = sentence[1][0]
    assert len(tags) == len(tokens)
    text.extend(tokens)
    for idx, tag in enumerate(tags):
        if tag != 'O':
            start = len(' '.join(tokens[:idx]))
            end = len(' '.join(tokens[:idx+1]))
            ents.append({"start":start, "end":end, "label":tag})
    return ' '.join(text), ents

if st.button('run'):
    # check if url is url or sentence
    if len(re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', url)) > 0:
        # it's a url

        html = requests.get(url).text

        text = fulltext(html)
    else:
        text = url

    output = process(text)

    # st.markdown(output)

    trigger_count = 0

    for docid, doc in output.items():
        st.markdown(f'## Document #{docid}')

        for sentid, sentence in enumerate(doc):

            # st.markdown(f'### Sentence #{sentid}')

            sentence_text, sentence_ents = align(sentence)

            if len(sentence_ents) > 0:
                trigger_count += 1

                st.markdown(f'### Sentence #{sentid}')

                ex = [{"text": sentence_text, "ents": sentence_ents, "title": None}]

                html = displacy.render(ex, style="ent", manual=True, minify=True, page=False)

                st.markdown(html, unsafe_allow_html=True)
            else:
                # st.markdown('Nothing found in this sentence.')
                pass

    # TODO - based on trigger_count, say if doc is about protest events or not