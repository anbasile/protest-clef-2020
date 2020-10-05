protesta fit tagger bert-base-cased protesta-data/task3
protesta fit tagger bert-base-cased protesta-data/task3 --crf-decoding
protesta fit tagger protest-model protesta-data/task3
protesta fit tagger protest-model protesta-data/task3 --crf-decoding

protesta predict outputs/tagger_protest-model_False Task3_train_dev_test/china_test.data
protesta predict outputs/tagger_protest-model_True Task3_train_dev_test/china_test.data
protesta predict outputs/tagger_bert-base-cased_False Task3_train_dev_test/china_test.data
protesta predict outputs/tagger_bert-base-cased_True Task3_train_dev_test/china_test.data

protesta predict outputs/tagger_protest-model_False protesta-data/task3/test.tsv
protesta predict outputs/tagger_protest-model_True protesta-data/task3/test.tsv
protesta predict outputs/tagger_bert-base-cased_False protesta-data/task3/test.tsv
protesta predict outputs/tagger_bert-base-cased_True protesta-data/task3/test.tsv

tar -cvf predictions.tar.gz protesta-data/task3/test.tagger* Task3_train_dev_test/china_test.tagger*