from enum import Enum


class ModelType(str, Enum):
    SequenceClassifier = "classifier"
    SequenceTagger = "tagger"


class EncodingMode(str, Enum):
    DocumentWise = "document"
    SentenceWise = "sentence"
