for model in bert-base-uncased 1-epoch 20-epoch 40-epoch 60-epoch 80-epoch 100-epoch
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
            tar xvf /content/drive/MyDrive/protest-predictions-and-models_bert_retrain_epochs/tagger_${model}_${decoding_boolean}_${mode}_1.0.tar.gz -C outputs/
            mv outputs/outputs/tagger_${model}_${decoding_boolean}_${mode}_1.0 outputs/tagger_${model}_${decoding_boolean}_${mode}_1.0
            protesta predict outputs/tagger_${model}_${decoding_boolean}_${mode}_1.0 /content/drive/My\ Drive/protesta-data/task3/dev.data
            mv /content/drive/MyDrive/protesta-data/task3/dev.tagger_${model}_${decoding_boolean}_${mode}_1.0 ./dev.predict
            zip /content/drive/MyDrive/protest-predictions-and-models_bert_retrain_epochs/dev_${model}_${decoding_boolean}_${mode}.zip ./dev.predict
            rm -rf outputs/
    done
    done
done
