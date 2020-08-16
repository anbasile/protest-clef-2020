from enum import Enum

class ModelType(str, Enum):
    SequenceClassifier = "classifier"
    SequenceTagger = "tagger"

