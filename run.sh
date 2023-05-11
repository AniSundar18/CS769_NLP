#Seq2Emo Baseline
python3 -u trainer_lstm_seq2emo.py --dataset goemotions --batch_size 64 --glove_path data/glove.840B.300d.txt --download_elmo --seed 0 --log_path logs/elmo.txt --output_path p_models/elmo.pt

#BERT-base baseline
#python3 -u trainer_lstm_seq2emo.py --dataset goemotions --en_dim 768 --batch_size 64 --encoder_model BERT --glove_path data/glove.840B.300d.txt --download_elmo --seed 0 --transformer_type base --log_path logs/bert.txt --output_path p_models/bert.pt

#SentiBERT baseline
#python3 -u trainer_lstm_seq2emo.py --dataset goemotions --en_dim 768 --batch_size 128 --encoder_model BERT --glove_path data/glove.840B.300d.txt --download_elmo --seed 0 --transformer_type "SentiBERT"   --log_path logs/sentibert.txt --output_path p_models/sentibert.pt

#Cardiff-Emoji
#python3 -u trainer_lstm_seq2emo.py --dataset goemotions --batch_size 64 --encoder_model RoBERTa --glove_path data/glove.840B.300d.txt --download_elmo --seed 0 --transformer_type "cardiff-emoji" --log_path logs/cardiff.txt --output_path p_models/cardiff.pt


#RoBERTa-base
#python3 -u trainer_lstm_seq2emo.py --dataset goemotions --batch_size 64 --encoder_model RoBERTa --glove_path data/glove.840B.300d.txt --download_elmo --seed 0 --transformer_type "base" --log_path logs/roberta_base.txt --output_path p_models/rob_base.pt

#RoBERTa-large
# python3 -u trainer_lstm_seq2emo.py --dataset goemotions --batch_size 64 --encoder_model RoBERTa --glove_path data/glove.840B.300d.txt --download_elmo --seed 0 --transformer_type "large" --log_path logs/roberta_large.txt --output_path p_models/rob_large.pt

#Active learning with LLM
#python3 -u trainer_lstm_seq2emo_active.py --dataset goemotions --batch_size 128 --glove_path data/glove.840B.300d.txt --download_elmo --model_save_path <YOUR SAVE PATH> --log_path <YOUR LOG FILE SAVE PATH> --split_mode 20k --openai_api_key <YOUR API KEY> --openai_org_key <YOUR ORG KEY> --use_LLM

#Evaluate model
#python3 -u evaluate.py --model_path <PATH TO SAVED MODEL> --download_elmo
