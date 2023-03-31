# Seq2Emo: A Sequence to Multi-Label Emotion Classification

This repository offers the script and environment settings needed for replicating the ablations we performed on the GoEmotions dataset and which are detailed in the project report.


## Requirements
The dependent packages are listed in the `requirements.txt`. Note that a cuda device is required.

We have included the data and preprocessing script under `data` folder.

We use [GloVe 840B 300d](https://nlp.stanford.edu/projects/glove/) pretrain embedding, you can extract and put the txt file under ``data`` folder.
or set the python argument `--glove_path` to point at it.  
You also need to use the argument  `--download_elmo` to download the ELMo embedding for first time of running the code.

The ELMo embeddings are needed for reproducing the results of the baseline.

## Training/evaluation 
The training results of various models are available in the logs folder.

For Seq2Emo (baseline), you can get the classification result of GoEmotions dataset by the following script.   

```
python3 -u trainer_lstm_seq2emo.py --dataset goemotions --batch_size 32 --glove_path data/glove.840B.300d.txt --download_elmo --seed 0 --log_path "YOUR FILE HERE"
```

To train the baseline, with BERT-base representations instead of ELMo,
```
python3 -u trainer_lstm_seq2emo.py --dataset goemotions  --batch_size 32 --encoder_model BERT --glove_path data/glove.840B.300d.txt --download_elmo --seed 0 --log_path "YOUR FILE HERE"
```

To train the baseline, with RoBERTa-base representations instead of ELMo,
```
python3 -u trainer_lstm_seq2emo.py --dataset goemotions  --batch_size 32 --encoder_model RoBERTa --glove_path data/glove.840B.300d.txt --download_elmo --seed 0 --transformer_type base --log_path "YOUR FILE HERE"
```

To train the baseline, with RoBERTa-large representations instead of ELMo,
```
python3 -u trainer_lstm_seq2emo.py --dataset goemotions  --batch_size 32 --encoder_model RoBERTa --glove_path data/glove.840B.300d.txt --download_elmo --seed 0 --transformer_type large --log_path "YOUR FILE HERE"
```

The same is available as a script in the file **run.sh**

Our changes to the code mainly exist in the following files,
```
trainer_lstm_seq2emo.py: Modified the code to include various kinds of text encoders.
models/seq2seq_lstm.py: Minor modifications in order to conduct experiments.
models/seq2seq_skep.py: Experimenting with certain concepts, not completely done yet.
```



