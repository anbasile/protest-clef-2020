import csv
import json
import os

import nlp


class ProtestConfig(nlp.BuilderConfig):
    """
        TODO
    """

    def __init__(self, features, **kwargs):
        super(ProtestConfig, self).__init__(
            version=nlp.Version("0.1.0"), **kwargs)
        self.features = features


class Protest(nlp.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        ProtestConfig(
            name="task1",
            features=["document", "label"],
        ),
        ProtestConfig(
            name="task2",
            features=["sentence", "label"],
        ),
        ProtestConfig(
            name="task3",
            features=["token", "label"],
        )]

    def _info(self):
        features = {feature: nlp.Value("string")
                    for feature in self.config.features}
        if self.config.name == 'task1':
            features["id"] = nlp.Value("int64")
            features["text"] = nlp.Value("string")
            features["url"] = nlp.Value("string")
            features["label"] = nlp.ClassLabel(names=["0", "1"])
        elif self.config.name == 'task2':
            features["id"] = nlp.Value("int64")
            features["label"] = nlp.ClassLabel(names=["0", "1"])
            features["last"] = nlp.Value("bool")
            features["sent_num"] = nlp.Value("int64")
            features["sentence"] = nlp.Value("string")
        elif self.config.name == 'task3':
            features['token'] = nlp.Sequence(nlp.Value("string"))
            features['label'] = nlp.Sequence(nlp.ClassLabel(names=[
                'B-etime',
                'B-fname',
                'B-loc',
                'B-organizer',
                'B-participant',
                'B-place',
                'B-target',
                'B-trigger',
                'I-etime',
                'I-fname',
                'I-loc',
                'I-organizer',
                'I-participant',
                'I-place',
                'I-target',
                'I-trigger',
                'O']))
        return nlp.DatasetInfo(
            features=nlp.Features(features),
        )

    def _split_generators(self, dl_manager):
        """ The `datafiles` kwarg in load_dataset() can be a str, List[str], Dict[str,str], or Dict[str,List[str]].
            If str or List[str], then the dataset returns only the 'train' split.
            If dict, then keys should be from the `nlp.Split` enum.
        """
        data_dir = self.config.data_dir
        extension_ = {
            'task1': 'jsonl',
            'task2': 'jsonl',
            'task3': 'tsv',
        }
        return [
            nlp.SplitGenerator(
                name=nlp.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"train.{extension_[self.config.name]}"),
                    # 'labelpath': os.path.join(data_dir, 'train_{}-labels.lst'.format(self.config.data_size)),
                    "split": "train",
                },
            ),
            nlp.SplitGenerator(
                name=nlp.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(
                    data_dir, f"test.{extension_[self.config.name]}"), "split": "test"},
            ),
            nlp.SplitGenerator(
                name=nlp.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"dev.{extension_[self.config.name]}"),
                    # 'labelpath': os.path.join(data_dir, 'dev-labels.lst'),
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath, split=None):
        """ Read files sequentially, then lines sequentially. """
        if self.config.name == 'task1':
            with open(filepath) as f:
                for id_, row in enumerate(f):
                    data = json.loads(row)
                    if split == 'test':
                        yield id_, {
                            "id": data['id'],
                            "label": -1,
                            "url": data['url'],
                            "text": data['text']
                        }
                    else:
                        yield id_, {
                            "id": data['id'],
                            "label": data['label'],
                            "url": data['url'],
                            "text": data['text']
                        }

        elif self.config.name == 'task2':
            with open(filepath) as f:
                for id_, row in enumerate(f):
                    data = json.loads(row)
                    if split == 'test':
                        yield id_, {
                            "id": data['id'],
                            "label": -1,
                            "last": data['last'],
                            "sent_num": data['sent_num'],
                            "sentence": data['sentence']
                        }
                    else:
                        yield id_, {
                            "id": data['id'],
                            "label": data['label'],
                            "last": data['last'],
                            "sent_num": data['sent_num'],
                            "sentence": data['sentence']
                        }
        elif self.config.name == 'task3':
            if split == 'test':
                with open(filepath, 'r', encoding='utf-8') as f:
                    sentence_index = 0
                    tokens = []
                    for row in f:
                        token = row.strip()
                        if token in ['SAMPLE_START', '[SEP]']:
                            continue
                        elif token == '':
                            yield sentence_index, {'token': tokens, 'label': []}
                            sentence_index += 1
                            tokens = []
                        else:
                            tokens.append(token)
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = csv.reader(f, delimiter='\t', quotechar=None)
                    sentence_index = 0
                    tokens = []
                    tags = []

                    for row in data:
                        try:
                            token, tag = row
                        except ValueError:  # empty row
                            sentence_index += 1
                            assert len(tokens) == len(tags)
                            yield sentence_index, {'token': tokens, 'label': tags}
                            tokens, tags = [], []  # reset for next sentence
                            continue
                        if token in ['SAMPLE_START', '[SEP]']:
                            continue
                        else:
                            tokens.append(token)
                            tags.append(tag)
