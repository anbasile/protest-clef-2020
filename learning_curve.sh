for model in bert-base-uncased protest-model
do
    for decoding in --crf-decoding
    do 
        for mode in sentence
        do
            for datasize in 0.1 0.3 0.5 0.7 0.9
            do
            protesta fit $model /content/drive/My\ Drive/protesta-data/task3/ --model-type=tagger $decoding --encoding=$mode --data-size=$datasize
            if [ $decoding = "--crf-decoding" ]; then
                decoding_boolean="True"
            else
                decoding_boolean="False"
            fi
            protesta predict outputs/tagger_${model}_${decoding_boolean}_${mode}_${datasize} /content/drive/My\ Drive/protesta-data/task3/test.data
            protesta predict outputs/tagger_${model}_${decoding_boolean}_${mode}_${datasize} /content/drive/My\ Drive/protesta-data/task3/china_test.data
            mv /content/drive/My\ Drive/protesta-data/task3/test.tagger_${model}_${decoding_boolean}_${mode}_${datasize} ./task3_test.predict
            mv /content/drive/My\ Drive/protesta-data/task3/china_test.tagger_${model}_${decoding_boolean}_${mode}_${datasize} ./china_test.predict
            zip /content/drive/My\ Drive/protest-predictions-and-models/${model}_${decoding_boolean}_${mode}_${datasize}.zip ./task3_test.predict ./china_test.predict
            tar -cvf /content/drive/My\ Drive/protest-predictions-and-models/tagger_${model}_${decoding_boolean}_${mode}_${datasize}.tar.gz outputs/tagger_${model}_${decoding_boolean}_${mode}_${datasize}
            rm -rf outputs/ task3_test.predict ./china_test.predict
    done
    done
    done
done
