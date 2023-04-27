#Seq2Emo Baseline
#python3 -u trainer_lstm_seq2emo.py --dataset goemotions --batch_size 32 --glove_path data/glove.840B.300d.txt --download_elmo --seed 0 
#SKEP encoder
#python3 -u trainer_lstm_seq2emo.py --dataset goemotions --en_dim 1024 --batch_size 32 --encoder_model BERT --glove_path data/glove.840B.300d.txt --download_elmo --seed 0 --transformer_type large --log_path logs/bert_large_log.txt
#python3 -u trainer_lstm_seq2emo.py --dataset goemotions --en_dim 768 --batch_size 128 --encoder_model BERT --glove_path data/glove.840B.300d.txt --download_elmo --seed 0 --transformer_type "SentiBERT"  --log_path logs/sentibert_log.txt 
python3 -u trainer_lstm_seq2emo.py --dataset goemotions --batch_size 64 --encoder_model RoBERTa --glove_path data/glove.840B.300d.txt --download_elmo --seed 0 --transformer_type "cardiff-emoji"

#--log_path logs/skep_log.txt
