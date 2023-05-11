# Investigating Contextual Representations for fine-grained Emotion Classification

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
python3 -u trainer_lstm_seq2emo.py --dataset goemotions --batch_size 64 --glove_path data/glove.840B.300d.txt --download_elmo --seed 0 --log_path "YOUR FILE HERE" --output_path "YOUR FILE HERE"
```

To train the baseline, with BERT-base representations instead of ELMo,
```
python3 -u trainer_lstm_seq2emo.py --dataset goemotions  --batch_size 32 --encoder_model BERT --glove_path data/glove.840B.300d.txt --download_elmo --seed 0 --log_path "YOUR FILE HERE" --output_path "YOUR FILE HERE"
```
To train the baseline, with SentiBERT representations instead of ELMo,
```
python3 -u trainer_lstm_seq2emo.py --dataset goemotions --en_dim 768 --batch_size 128 --encoder_model BERT --glove_path data/glove.840B.300d.txt --download_elmo --seed 0 --transformer_type "SentiBERT"  --log_path "YOUR FILE HERE" --output_path "YOUR FILE HERE"
```

To train the baseline, with RoBERTa-base representations instead of ELMo,
```
python3 -u trainer_lstm_seq2emo.py --dataset goemotions --batch_size 64 --encoder_model RoBERTa --glove_path data/glove.840B.300d.txt --download_elmo --seed 0 --transformer_type "base" --log_path "YOUR FILE HERE" --output_path "YOUR FILE HERE"
```

To train the baseline, with RoBERTa-large representations instead of ELMo,
```
python3 -u trainer_lstm_seq2emo.py --dataset goemotions --batch_size 64 --encoder_model RoBERTa --glove_path data/glove.840B.300d.txt --download_elmo --seed 0 --transformer_type "large" --log_path "YOUR FILE HERE" --output_path "YOUR FILE HERE"
```
To train the baseline, with RoBERTa-Cardiff-Emoji representations instead of ELMo,
```
python3 -u trainer_lstm_seq2emo.py --dataset goemotions --batch_size 64 --encoder_model RoBERTa --glove_path data/glove.840B.300d.txt --download_elmo --seed 0 --transformer_type "cardiff-emoji" --log_path "YOUR FILE HERE" --output_path "YOUR FILE HERE"
```
The same is available as a script in the file **run.sh**

Our changes to the code mainly exist in the following files,

* **trainer_lstm_seq2emo.py**: Modified the code to include various kinds of text encoders.
* **models/seq2seq_lstm.py**: Minor modifications in order to conduct experiments.
* **models/seq2seq_skep.py**: Experimenting with certain concepts, not completely done yet.

# Investigating LLM-based Active Learning

## Training:

We perform active learning experiments only on the base Seq2Emo model due to GPU and time constraints. An example script to invoke the active learning loop (without LLM annotation) would be as given below:
```
python3 -u trainer_lstm_seq2emo_active.py --dataset goemotions --batch_size 128 --glove_path data/glove.840B.300d.txt --download_elmo --model_save_path < YOUR SAVE PATH> --log_path <YOUR LOG FILE SAVE PATH> --split_mode 20k 
--split_mode decides how many training examples you will have in your X_train
```
This will sample 5k samples points from the unlabeled set and use the ground truth target labels as the annotation in a simulated active learning environment
```
python3 -u trainer_lstm_seq2emo_active.py --dataset goemotions --batch_size 128 --glove_path data/glove.840B.300d.txt --download_elmo --model_save_path < YOUR SAVE PATH> --log_path <YOUR LOG FILE SAVE PATH> --split_mode 20k --openai_api_key <YOUR API KEY> --openai_org_key <YOUR ORG KEY> --use_LLM
```
Since we have used GPT-3.5-turbo from OpenAI API as our annotator, you will require an OpenAI api and org key to run the LLM annotation active learning loop. The `--use_LLM` argument is just a flag that lets the script know that you will be using the LLM annotations.

## Evaluation:

For evaluation, we write our own evaluation script which can be run as given below:
```
python3 -u evaluate.py  --model_path <PATH TO SAVED MODEL> --download_elmo
```
You can download our saved checkpoints from [here](https://drive.google.com/drive/folders/1YxL6qHy_iLkGA0PbfNxI65bdDzDxD3z7?usp=sharing) and see the results as in the paper. For reference (logs for the training runs are under the folder `logs/AL/`):

`seq2emo_NoAL_20k.pt`, `seq2emo_NoAL_25k.pt`, `seq2emo_NoAL_30k.pt`: These correspond to models training for 10 epochs without Active learning with 20k, 25k and 30k data points from the training set respectively. Corresponding log files: `no_AL_20k.txt`, `no_AL_25k.txt`, `no_AL_30k.txt`

`seq2emo_25K_NoLLM.pt`: This corresponds to the model trained with active learning in a simulated environment with annotations from the target label set instead of the LLM annotations. Corresponding log file: `AL_25k_NoLLM.txt`

`seq2emo_25K_LLM.pt`: This corresponds to the model trained with active learning with LLM annotation feedback. Corresponding log file: `AL_25k_LLM.txt`




