# protesta fit tagger bert-base-cased protesta-data/task3
# do
# protesta fit tagger bert-base-cased protesta-data/task3 --crf-decoding
# do
# protesta fit tagger protest-model protesta-data/task3
# protesta fit tagger protest-model protesta-data/task3 --crf-decoding
# 
# protesta predict outputs/tagger_protest-model_False Task3_train_dev_test/china_test.data
# protesta predict outputs/tagger_protest-model_True Task3_train_dev_test/china_test.data
# protesta predict outputs/tagger_bert-base-cased_False Task3_train_dev_test/china_test.data
# protesta predict outputs/tagger_bert-base-cased_True Task3_train_dev_test/china_test.data
# 
# protesta predict outputs/tagger_protest-model_False protesta-data/task3/test.tsv
# protesta predict outputs/tagger_protest-model_True protesta-data/task3/test.tsv
# protesta predict outputs/tagger_bert-base-cased_False protesta-data/task3/test.tsv
# protesta predict outputs/tagger_bert-base-cased_True protesta-data/task3/test.tsv
# 
# tar -cvf predictions.tar.gz protesta-data/task3/test.tagger* Task3_train_dev_test/china_test.tagger*
# !protesta fit tagger bert-base-uncased /content/drive/My\ Drive/protesta-data/task3/ document
# !protesta predict outputs/tagger_bert-base-uncased_True /content/drive/My\ Drive/protesta-data/task3/test.tsv
# !protesta predict outputs/tagger_bert-base-uncased_True /content/drive/My\ Drive/protesta-data/task3/china_test.data
for model in bert-base-uncased protest-bert
do
    for decoding in --no-crf-decoding --crf-decoding
    do 
        for mode in document sentence
        do
        protesta fit tagger $model /content/drive/My\ Drive/protesta-data/task3/ $decoding $mode
        if [ $decoding = "--crf-decoding" ]; then
            protesta predict outputs/${model}_True /content/drive/My\ Drive/protesta-data/task3/test.tsv
            protesta predict outputs/${model}_True /content/drive/My\ Drive/protesta-data/task3/china_test.data
        else
            protesta predict outputs/${model}_False /content/drive/My\ Drive/protesta-data/task3/test.tsv
            protesta predict outputs/${model}_False /content/drive/My\ Drive/protesta-data/task3/china_test.data
        fi
    done
    done
done