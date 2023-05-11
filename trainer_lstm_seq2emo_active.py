import os
import random
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from models.seq2seq_lstm import LSTMSeq2Seq

import torch.nn as nn
from transformers import RobertaTokenizer
from transformers import AutoTokenizer, RobertaModel
from transformers import BertTokenizer, BertModel
import torch.optim as optim
from tqdm import tqdm
from utils.early_stopping import EarlyStopping
import pickle as pkl
from utils.seq2emo_metric import get_metrics, get_multi_metrics, jaccard_score, get_single_metrics
from utils.tokenizer import GloveTokenizer
from copy import deepcopy
from allennlp.modules.elmo import Elmo, batch_to_ids
import argparse
from data.data_loader import load_sem18_data, load_goemotions_data
from utils.scheduler import get_cosine_schedule_with_warmup
import utils.nn_utils as nn_utils
from utils.others import find_majority
from utils.file_logger import get_file_logger
import pickle
import math
import copy
import json
from datetime import datetime
import openai
import ast
import time

# Argument parser
parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--batch_size', default=32, type=int, help="batch size")
parser.add_argument('--pad_len', default=50, type=int, help="batch size")
parser.add_argument('--postname', default='', type=str, help="post name")
parser.add_argument('--gamma', default=0.2, type=float, help="post name")
parser.add_argument('--folds', default=5, type=int, help="num of folds")
parser.add_argument('--en_lr', default=5e-4, type=float, help="encoder learning rate")
parser.add_argument('--de_lr', default=1e-4, type=float, help="decoder learning rate")
parser.add_argument('--loss', default='ce', type=str, help="loss function ce/focal")
parser.add_argument('--dataset', default='sem18', type=str, choices=['sem18', 'goemotions'])
parser.add_argument('--en_dim', default=1200, type=int, help="dimension")
parser.add_argument('--de_dim', default=400, type=int, help="dimension")
parser.add_argument('--criterion', default='jaccard', type=str, choices=['jaccard', 'macro', 'micro', 'h_loss'])
parser.add_argument('--glove_path', default='data/glove.840B.300d.txt', type=str)
parser.add_argument('--attention', default='dot', type=str, help='general/mlp/dot')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout rate')
parser.add_argument('--encoder_dropout', default=0.2, type=float, help='dropout rate')
parser.add_argument('--decoder_dropout', default=0, type=float, help='dropout rate')
parser.add_argument('--attention_dropout', default=0.2, type=float, help='dropout rate')
parser.add_argument('--patience', default=13, type=int, help='dropout rate')
parser.add_argument('--download_elmo', action='store_true')
parser.add_argument('--scheduler', action='store_true')
parser.add_argument('--glorot_init', action='store_true')
parser.add_argument('--warmup_epoch', default=0, type=float, help='')
parser.add_argument('--stop_epoch', default=10, type=float, help='')
parser.add_argument('--max_epoch', default=20, type=float, help='')
parser.add_argument('--min_lr_ratio', default=0.1, type=float, help='')
parser.add_argument('--fix_emb', action='store_true')
parser.add_argument('--fix_emo_emb', action='store_true')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--input_feeding', action='store_true', default=True)
parser.add_argument('--dev_split_seed', type=int, default=0)
parser.add_argument('--normal_init', action='store_true')
parser.add_argument('--unify_decoder', action='store_true')
parser.add_argument('--eval_every', type=bool, default=True)
parser.add_argument('--log_path', type=str, default=None)
parser.add_argument('--attention_heads', type=int, default=1)
parser.add_argument('--concat_signal', action='store_true')
parser.add_argument('--no_cross', action='store_true')
parser.add_argument('--output_path', type=str, default=None)
parser.add_argument('--attention_type', type=str, default="luong", choices=['transformer', 'luong'])
parser.add_argument('--load_emo_emb', action='store_true')
parser.add_argument('--shuffle_emo', type=str, default=None)
parser.add_argument('--single_direction', action='store_true')
parser.add_argument('--encoder_model', type=str, default='LSTM')
parser.add_argument('--transformer_type', type=str, default='base')
parser.add_argument('--model_save_path', type=str, default=None)
parser.add_argument('--openai_org_key', type=str, default=None)
parser.add_argument('--openai_api_key', type=str, default=None)
parser.add_argument('--use_LLM', action='store_true')


args = parser.parse_args()
openai.organization = args.openai_org_key
openai.api_key = args.openai_api_key
with open('prompt.txt', 'r') as f:
    prompt = f.read()
if args.log_path is not None:
    dir_path = os.path.dirname(args.log_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

logger = get_file_logger(args.log_path)  # Note: this is ugly, but I am lazy

SRC_EMB_DIM = 300
MAX_LEN_DATA = args.pad_len
PAD_LEN = MAX_LEN_DATA
MIN_LEN_DATA = 3
BATCH_SIZE = args.batch_size
CLIPS = 0.666
GAMMA = 0.5
SRC_HIDDEN_DIM = args.en_dim
TGT_HIDDEN_DIM = args.de_dim
VOCAB_SIZE = 60000
ENCODER_LEARNING_RATE = args.en_lr
DECODER_LEARNING_RATE = args.de_lr
ATTENTION = args.attention
PATIENCE = args.patience
WARMUP_EPOCH = args.warmup_epoch
STOP_EPOCH = int(args.stop_epoch)
MAX_EPOCH = int(args.max_epoch)
RANDOM_SEED = args.seed
# Seed
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)



# Init Elmo model
if True:
    if args.download_elmo:
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    else:
        options_file = "elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    elmo = Elmo(options_file, weight_file, 2, dropout=0).cuda()
    elmo.eval()

    GLOVE_EMB_PATH = args.glove_path
    glove_tokenizer = GloveTokenizer(PAD_LEN)

    data_path_postfix = '_split'
    data_pkl_path = 'data/' + args.dataset + data_path_postfix + '_data.pkl'
    if not os.path.isfile(data_pkl_path):
        if args.dataset == 'sem18':
            X_train_dev, y_train_dev, X_test, y_test, EMOS, EMOS_DIC, data_set_name = \
                load_sem18_data()
        elif args.dataset == 'goemotions':
            X_train_dev, y_train_dev, X_test, y_test, EMOS, EMOS_DIC, data_set_name = \
                load_goemotions_data()
        else:
            raise NotImplementedError

        with open(data_pkl_path, 'wb') as f:
            logger('Writing file')
            pkl.dump((X_train_dev, y_train_dev, X_test, y_test, EMOS, EMOS_DIC, data_set_name), f)

    else:
        with open(data_pkl_path, 'rb') as f:
            logger('loading file')
            X_train_dev, y_train_dev, X_test, y_test, EMOS, EMOS_DIC, data_set_name = pkl.load(f)

if args.encoder_model == 'SKEP':
    skep_tokenizer = RobertaTokenizer.from_pretrained('Yaxin/roberta-large-ernie2-skep-en', padding='max_length', truncation=True, max_length=42)
    print("Token: ", skep_tokenizer.pad_token)
    data_path_postfix = '_split'
    data_pkl_path = 'data/' + args.dataset + data_path_postfix + '_data.pkl'
    if not os.path.isfile(data_pkl_path):
        if args.dataset == 'sem18':
            X_train_dev, y_train_dev, X_test, y_test, EMOS, EMOS_DIC, data_set_name = \
                load_sem18_data()
        elif args.dataset == 'goemotions':
            X_train_dev, y_train_dev, X_test, y_test, EMOS, EMOS_DIC, data_set_name = \
                load_goemotions_data()
        else:
            raise NotImplementedError

        with open(data_pkl_path, 'wb') as f:
            logger('Writing file')
            pkl.dump((X_train_dev, y_train_dev, X_test, y_test, EMOS, EMOS_DIC, data_set_name), f)

    else:
        with open(data_pkl_path, 'rb') as f:
            logger('loading file')
            X_train_dev, y_train_dev, X_test, y_test, EMOS, EMOS_DIC, data_set_name = pkl.load(f)

elif args.encoder_model == 'BERT':
    if args.transformer_type == 'base':
        model_name = 'bert-base-uncased'
        tokenizer_name = 'bert-base-cased'
    elif args.transformer_type=='large':
        model_name = 'bert-large-uncased'
        tokenizer_name = 'bert-large-uncased'
    elif args.transformer_type == 'SentiBERT':
        model_name = 'adresgezgini/Finetuned-SentiBERtr-Pos-Neg-Reviews'
        tokenizer_name = 'adresgezgini/Finetuned-SentiBERtr-Pos-Neg-Reviews'
    bert_tokenizer = BertTokenizer.from_pretrained(model_name, padding = 'max_len', max_length=42, truncation = True)
    bert_model = BertModel.from_pretrained(tokenizer_name)

elif args.encoder_model == 'RoBERTa':
    if args.transformer_type == 'base':
        model_type = 'roberta-base'
    elif args.transformer_type == 'cardiff-emoji':
        model_type = 'cardiffnlp/twitter-roberta-base-emoji'
    elif args.transformer_type == 'large':
        model_type = 'roberta-large'
    roberta_tokenizer = AutoTokenizer.from_pretrained(model_type)
    roberta_model = RobertaModel.from_pretrained(model_type)


NUM_EMO = len(EMOS)


class TestDataReader(Dataset):
    def __init__(self, X, pad_len):
        self.glove_ids = []
        self.glove_ids_len = []
        self.pad_len = pad_len
        self.build_glove_ids(X)

    def build_glove_ids(self, X):
        for src in X:
            glove_id, glove_id_len = glove_tokenizer.encode_ids_pad(src)
            self.glove_ids.append(glove_id)
            self.glove_ids_len.append(glove_id_len)

    def __len__(self):
        return len(self.glove_ids)

    def __getitem__(self, idx):
        return torch.LongTensor(self.glove_ids[idx]), \
               torch.LongTensor([self.glove_ids_len[idx]])


class TrainDataReader(TestDataReader):
    def __init__(self, X, y, pad_len):
        super(TrainDataReader, self).__init__(X, pad_len)
        self.y = []
        self.read_target(y)
        self.X = X

    def read_target(self, y):
        self.y = y
    def __getitem__(self, idx):
        return torch.LongTensor(self.glove_ids[idx]), \
               torch.LongTensor([self.glove_ids_len[idx]]), \
               torch.LongTensor(self.y[idx]), self.X[idx]


def elmo_encode(ids):
    data_text = [glove_tokenizer.decode_ids(x) for x in ids]
    with torch.no_grad():
        character_ids = batch_to_ids(data_text).cuda()
        elmo_emb = elmo(character_ids)['elmo_representations']
        elmo_emb = (elmo_emb[0] + elmo_emb[1]) / 2  # avg of two layers
    return elmo_emb

def bert_encode(ids):
    data_text = [" ".join(glove_tokenizer.decode_ids(x)) for x in ids]
    with torch.no_grad():
        character_ids = bert_tokenizer(data_text, return_tensors='pt', padding='max_length', truncation = True, max_length=50)
        bert_emb = bert_model(**character_ids).last_hidden_state
    return bert_emb

def roberta_encode(ids):
    data_text = [" ".join(glove_tokenizer.decode_ids(x)) for x in ids]
    with torch.no_grad():
        character_ids = roberta_tokenizer(data_text, return_tensors='pt', padding='max_length', truncation = True, max_length=50)
        roberta_emb = roberta_model(**character_ids).last_hidden_state
    return roberta_emb

def show_classification_report(gold, pred):
    from sklearn.metrics import classification_report
    logger(classification_report(gold, pred, target_names=EMOS, digits=4))


def eval(model, best_model, loss_criterion, es, dev_loader, dev_set, dev_mode=True):
    # Evaluate
    
    exit_training = False
    model.eval()
    test_loss_sum = 0
    preds = []
    gold = []
    d_preds = []
    X_pool = []
    y_pool = []
    logger("Evaluating:")
    for i, loader_input in tqdm(enumerate(dev_loader), total=int(len(dev_set) / BATCH_SIZE), disable=True):
        src, src_len, trg, nl = loader_input
        with torch.no_grad():
            if args.encoder_model == 'LSTM':
                elmo_src = elmo_encode(src)
                decoder_logit = model(src.cuda(), src_len.cuda(), elmo_src.cuda())
            elif args.encoder_model == 'BERT':
                bert_src = bert_encode(src)
                decoder_logit = model(src.cuda(), src_len.cuda(), bert_src.cuda())
            elif args.encoder_model == 'RoBERTa':
                roberta_src = roberta_encode(src)
                decoder_logit = model(src.cuda(), src_len.cuda(), roberta_src.cuda())
            test_loss = loss_criterion(
                decoder_logit.view(-1, decoder_logit.shape[-1]),
                trg.view(-1).cuda()
            )
            test_loss_sum += test_loss.data.cpu().numpy() * src.shape[0]
            gold.append(trg.data.numpy())
            d_preds.append(decoder_logit.cpu().detach().numpy())
            X_pool.append(nl)
            y_pool.append(trg)
            preds.append(np.argmax(decoder_logit.data.cpu().numpy(), axis=-1))
            
            del decoder_logit
    if not dev_mode:
        return d_preds, X_pool, y_pool
    preds = np.concatenate(preds, axis=0)
    gold = np.concatenate(gold, axis=0)
    
    metric = get_metrics(gold, preds)
    jaccard = jaccard_score(gold, preds)
    logger("Evaluation results:")
    # show_classification_report(binary_gold, binary_preds)
    logger("Evaluation Loss", test_loss_sum / len(dev_set))

    logger('Normal: h_loss:', metric[0], 'macro F', metric[1], 'micro F', metric[4], 'micro P', metric[5],
          'micro R', metric[6])
    metric_2 = get_multi_metrics(gold, preds)
    logger('Multi only: h_loss:', metric_2[0], 'macro F', metric_2[1], 'micro F', metric_2[4])
    logger('Jaccard:', jaccard)

    if args.criterion == 'loss':
        criterion = test_loss_sum
    elif args.criterion == 'macro':
        criterion = 1 - metric[1]
    elif args.criterion == 'micro':
        criterion = 1 - metric[4]
    elif args.criterion == 'h_loss':
        criterion = metric[0]
    elif args.criterion == 'jaccard':
        criterion = 1 - jaccard
    else:
        raise ValueError

    if es.step(criterion):  # overfitting
        del model
        logger('overfitting, loading best model ...')
        model = best_model
        exit_training = True
    else:
        if es.is_best():
            if best_model is not None:
                del best_model
            logger('saving best model ...')
            best_model = deepcopy(model)
        else:
            logger(f'patience {es.cur_patience} not best model , ignoring ...')
            if best_model is None:
                best_model = deepcopy(model)

    return model, best_model, exit_training


def train(X_train, y_train, X_dev, y_dev, X_test, y_test, model=None, file_ext=''):
        

        # Model initialize

        train_set = TrainDataReader(X_train, y_train, MAX_LEN_DATA)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

        dev_set = TrainDataReader(X_dev, y_dev, MAX_LEN_DATA)
        dev_loader = DataLoader(dev_set, batch_size=BATCH_SIZE*3, shuffle=False)

        test_set = TestDataReader(X_test, MAX_LEN_DATA)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE*3, shuffle=False)
        if not model:
            model = LSTMSeq2Seq(
                emb_dim=SRC_EMB_DIM,
                vocab_size=glove_tokenizer.get_vocab_size(),
                trg_vocab_size=NUM_EMO,
                src_hidden_dim=SRC_HIDDEN_DIM,
                trg_hidden_dim=TGT_HIDDEN_DIM,
                attention_mode=ATTENTION,
                batch_size=BATCH_SIZE,
                nlayers=2,
                nlayers_trg=2,
                dropout=args.dropout,
                encoder_dropout=args.encoder_dropout,
                decoder_dropout=args.decoder_dropout,
                attention_dropout=args.attention_dropout,
                args=args
            )

        if args.fix_emb:
            para_group = [
                {'params': [p for n, p in model.named_parameters() if n.startswith("encoder") and
                            not 'encoder.embeddings' in n], 'lr': args.en_lr},
                {'params': [p for n, p in model.named_parameters() if n.startswith("decoder")], 'lr': args.de_lr}]
        else:
            para_group = [
                {'params': [p for n, p in model.named_parameters() if n.startswith("encoder")], 'lr': args.en_lr},
                {'params': [p for n, p in model.named_parameters() if n.startswith("decoder")], 'lr': args.de_lr}]
        loss_criterion = nn.CrossEntropyLoss() # reduction='sum'
        optimizer = optim.Adam(para_group)
        if args.scheduler:
            epoch_to_step = len(train_set) / BATCH_SIZE
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=int(WARMUP_EPOCH * epoch_to_step),
                num_training_steps=int(STOP_EPOCH * epoch_to_step),
                min_lr_ratio=args.min_lr_ratio
            )

        if args.glorot_init:
            logger('use glorot initialization')
            for group in para_group:
                nn_utils.glorot_init(group['params'])

        model.load_encoder_embedding(glove_tokenizer.get_embeddings(), fix_emb=args.fix_emb)
        model.load_emotion_embedding(glove_tokenizer.get_emb_by_words(GLOVE_EMB_PATH, EMOS))
        model.cuda()

        # Start training
        EVAL_EVERY = int(len(train_set) / BATCH_SIZE / 4)
        best_model = None
        es = EarlyStopping(patience=PATIENCE)
        update_step = 0
        exit_training = False
        for epoch in range(1, MAX_EPOCH+1):
            logger('Training on epoch=%d -------------------------' % (epoch))
            train_loss_sum = 0
                # print('Current encoder learning rate', scheduler.get_lr())
                # print('Current decoder learning rate', scheduler.get_lr())

            for i, loader_input in tqdm(enumerate(train_loader), total=int(len(train_set) / BATCH_SIZE)):
                model.train()
                update_step += 1
                # print('i=%d: ' % (i))
                # trg = torch.index_select(trg, 1, torch.LongTensor(list(range(1, len(EMOS)+1))))
                src, src_len, trg, _ = loader_input
                optimizer.zero_grad()
                if args.encoder_model == 'LSTM':
                    elmo_src = elmo_encode(src)
                    #print(elmo_src.shape, src.shape)
                    decoder_logit = model(src.cuda(), src_len.cuda(), elmo_src.cuda())
                elif args.encoder_model == 'BERT' :
                    bert_src = bert_encode(src)[:,:src.shape[1],:]
                    decoder_logit = model(src.cuda(), src_len.cuda(), bert_src.cuda())
                elif args.encoder_model == 'RoBERTa' :
                    roberta_src = roberta_encode(src)[:,:src.shape[1],:]
                    decoder_logit = model(src.cuda(), src_len.cuda(), roberta_src.cuda())

                loss = loss_criterion(
                    decoder_logit.view(-1, decoder_logit.shape[-1]),
                    trg.view(-1).cuda()
                )# /src.size(0)
                loss.backward()
                train_loss_sum += loss.data.cpu().numpy() * src.shape[0]

                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIPS)
                optimizer.step()
                if args.scheduler:
                    scheduler.step()

                if update_step % EVAL_EVERY == 0 and args.eval_every is not None:
                    model, best_model, exit_training = eval(model, best_model, loss_criterion, es, dev_loader, dev_set)
                    if exit_training:
                        break

            logger(f"Training Loss for epoch {epoch}:", train_loss_sum / len(train_set))
            model, best_model, exit_training = eval(model, best_model, loss_criterion, es, dev_loader, dev_set)
            if exit_training:
                break

        # final_testing
        model.eval()
        preds = []
        logger("Testing:")
        for i, loader_input in tqdm(enumerate(test_loader), total=int(len(test_set) / BATCH_SIZE)):
            with torch.no_grad():
                src, src_len = loader_input
                if args.encoder_model == "LSTM":
                    elmo_src = elmo_encode(src)
                    decoder_logit = model(src.cuda(), src_len.cuda(), elmo_src.cuda())
                    preds.append(np.argmax(decoder_logit.data.cpu().numpy(), axis=-1))
                    del decoder_logit
                elif args.encoder_model == 'BERT':
                    bert_src = bert_encode(src)
                    decoder_logit = model(src.cuda(), src_len.cuda(), bert_src.cuda())
                    preds.append(np.argmax(decoder_logit.data.cpu().numpy(), axis=-1))
                    del decoder_logit
                elif args.encoder_model == "RoBERTa":
                    roberta_src = roberta_encode(src)
                    decoder_logit = model(src.cuda(), src_len.cuda(), roberta_src.cuda())
                    preds.append(np.argmax(decoder_logit.data.cpu().numpy(), axis=-1))

        torch.save(best_model.state_dict(), args.model_save_path+str(file_ext)+'.pt')
        preds = np.concatenate(preds, axis=0)
        gold = np.asarray(y_test)
        binary_gold = gold
        binary_preds = preds
        logger("NOTE, this is on the test set")
        metric = get_metrics(binary_gold, binary_preds)
        logger('Normal: h_loss:', metric[0], 'macro F', metric[1], 'micro F', metric[4])
        metric = get_multi_metrics(binary_gold, binary_preds)
        logger('Multi only: h_loss:', metric[0], 'macro F', metric[1], 'micro F', metric[4])
        # show_classification_report(binary_gold, binary_preds)
        logger('Jaccard:', jaccard_score(gold, preds))
        return binary_gold, binary_preds, best_model


def main():
    global X_train_dev, X_test, y_train_dev, y_test

    glove_tokenizer.build_tokenizer(X_train_dev + X_test, vocab_size=VOCAB_SIZE)
    glove_tokenizer.build_embedding(GLOVE_EMB_PATH, dataset_name=data_set_name)
    X_train, y_train = X_train_dev[:30000], y_train_dev[:30000]
    X_pool, y_pool = X_train_dev[30000:40000], y_train_dev[30000:40000]
    X_dev, y_dev = X_train_dev[40000:], y_train_dev[40000:]

    all_preds = []
    gold_list = None
    reverse_emo = {}
    for e in EMOS_DIC:
        reverse_emo[EMOS_DIC[e]] = e

    def calculate_probs(x):
        probs = []
        for i in x:
            lasts = [p[-1] for p in i]
            probs.append(np.exp(lasts)/sum(np.exp(lasts)))
        return np.asarray(probs)

    def entropy_calc(prob_dist):
        log_probs = prob_dist * np.log2(prob_dist)
        raw_entropy = 0 - np.sum(log_probs)
        normalized_entropy = raw_entropy / math.log2(len(prob_dist))
        return normalized_entropy


    def list_check(l):
        for e in l:
            if e not in EMOS_DIC:
                return False
        return True

    def get_prompt_ops(nl, idx, y_pool):
        if args.use_LLM:
            #pass nl to openai api
            full_prompt = prompt+ '\n\n[Statement]: '+nl+'\n[Answer]:'
            #get result
            print('OpenAI Prompt!')
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                        {"role": "user", "content": full_prompt},
                    ]
                )
            #peform list check
            op = response['choices'][0]['message']['content']
            try:
                op_clean = ast.literal_eval(op)
                
            except SyntaxError:
                print("SyntaxError at index "+str(idx))
                print(op)
                op_clean = [reverse_emo[idx_1] for idx_1, i in enumerate(y_pool) if i]
            except ValueError:
                print("ValueError at index "+str(idx))
                print(op)
                op_clean = [reverse_emo[idx_1] for idx_1, i in enumerate(y_pool) if i]
            l_check = list_check(op_clean)
            if not l_check:
                print("L_check failed at index "+str(idx))
                print(op_clean)
                op_clean = [reverse_emo[idx_1] for idx_1, i in enumerate(y_pool) if i]
        else:
            response ='Not using LLM, Simulated mode'
            op_clean = [reverse_emo[idx_1] for idx_1, i in enumerate(y_pool) if i]
        emb = [0]*28
        for i in op_clean:
            emb[EMOS_DIC[i]] = 1

        #pass output in format as output list of the emotions, check variable
        return emb, response

    
    best_model = None
    loss_criterion = nn.CrossEntropyLoss()
    es = EarlyStopping(patience=PATIENCE)
    n = 500
    X_pool = np.asarray(X_pool)
    y_pool = np.asarray(y_pool)
    pool_done = {}
    done_els = []
    iterations = 10
    cache_list = []
    gold_list, pred_list, model = train(X_train, y_train, X_dev, y_dev, X_test, y_test)
    for idx_2 in range(iterations):
        pool_set = TrainDataReader(X_pool, y_pool, MAX_LEN_DATA)
        pool_loader = DataLoader(pool_set, batch_size=BATCH_SIZE*3, shuffle=False)
        d_preds, X_pool_new, y_pool_new = eval(model, best_model, loss_criterion, es, pool_loader, pool_set, dev_mode=False)
        preds = []
        for i in d_preds:
            preds.append(np.argmax(i, axis=-1))
        preds = np.concatenate(preds, axis=0)
        d_preds = np.concatenate(d_preds, axis=0)
        X_pool_new = np.asarray(X_pool_new)
        X_pool_new = np.concatenate(X_pool_new, axis=0)
        probs = calculate_probs(d_preds)
        entropies = []
        for p in probs:
            entropies.append(entropy_calc(p))
        pred_list = []
        for i in range(len(X_pool_new)):
            if i not in done_els:
                pred_list.append({'index':i,'sentence':X_pool_new[i],'entropy':entropies[i], 'd_preds': d_preds[i], 'preds':preds[i],'y_pool':y_pool[i]})
        sorted_pred_list = sorted(pred_list, key=lambda d: d['entropy'], reverse=True)
        top_n = sorted_pred_list[:n]
        pool_done = {}
        
        for t in top_n:
            y_pred_emb, response = get_prompt_ops(t['sentence'],idx_2, t['y_pool'])
            cache_list.append({'index':t['index'], 'iteration':idx_2, 'response':response})
            pool_done[t['index']] = {'sentence':X_pool_new[t['index']], 'prediction':y_pred_emb}
        #add pool_done elements to train set
        for p in pool_done:
            X_train.append(pool_done[p]['sentence'])
            y_train.append(pool_done[p]['prediction'])
        logger(str(len(X_train)))
        logger(str(len(y_train)))
        #train
        gold_list, pred_list, model = train(X_train, y_train, X_dev, y_dev, X_test, y_test, model, idx_2)
        all_preds.append(pred_list)
        if args.no_cross:
            break
        #delete from pool
        for val in pool_done:
            done_els.append(val)
    print('done')


if __name__ == '__main__':
    main()


