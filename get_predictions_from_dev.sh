for model in bert-base-uncased protest-model
do
    for decoding in --no-crf-decoding --crf-decoding
    do 
        for mode in sentence
        do
            mkdir outputs/
            if [ $decoding = "--crf-decoding" ]; then
                decoding_boolean="True"
            else
                decoding_boolean="False"
            fi
            tar xvf /content/drive/My\ Drive/protest-predictions-and-models_bert_retrain_epochs/tagger_${model}_${decoding_boolean}_${mode}_1.0.tar.gz -C outputs/
            mv outputs/outputs/tagger_${model}_${decoding_boolean}_${mode}_1.0 outputs/tagger_${model}_${decoding_boolean}_${mode}_1.0
            protesta predict outputs/tagger_${model}_${decoding_boolean}_${mode}_1.0 /content/drive/My\ Drive/protesta-data/task3/dev.data
            mv /content/drive/My\ Drive/protesta-data/task3/dev.tagger_${model}_${decoding_boolean}_${mode}_1.0 ./dev.predict
            zip /content/drive/My\ Drive/protest-predictions-and-models_bert_retrain_epochs/dev_${model}_${decoding_boolean}_${mode}.zip ./dev.predict
            rm -rf outputs/
    done
    done
done
