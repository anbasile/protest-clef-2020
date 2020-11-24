for model in bert-base-uncased protest-model
do
    for decoding in --no-crf-decoding --crf-decoding
    do 
        for mode in document sentence
        do
        protesta fit tagger $model /content/drive/My\ Drive/protesta-data/task3/ $decoding $mode
        if [ $decoding = "--crf-decoding" ]; then
            decoding_boolean="True"
        else
            decoding_boolean="False"
        fi
        protesta predict outputs/${model}_${decoding_boolean}_${mode} /content/drive/My\ Drive/protesta-data/task3/test.tsv
        protesta predict outputs/${model}_${decoding_boolean}_${mode} /content/drive/My\ Drive/protesta-data/task3/china_test.data
        mv /content/drive/My\ Drive/protesta-data/task3/test.tagger_${model}_${decoding_boolean}_${mode} ./task3_test.predict
        mv /content/drive/My\ Drive/protesta-data/task3/china_test.tagger_${model}_${decoding_boolean}_${mode} ./china_test.predict
        zip /content/drive/My\ Drive/protest-predictions-and-models/${model}_${decoding_boolean}_${mode}.zip ./task3_test.predict ./china_test.predict
        tar -cvf /content/drive/My\ Drive/protest-predictions-and-models/tagger_${model}_${decoding_boolean}_${mode}.tar.gz outputs/tagger_${model}_${decoding_boolean}_${mode}
        rm -rf outputs/ task3_test.predict ./china_test.predict
    done
    done
done
