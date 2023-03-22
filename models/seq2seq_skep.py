import torch
import torch.nn as nn
from modules.lstm_encoder import LSTMEncoder
from modules.seq2seq_decoder import Seq2SeqDecoder
from transformers import RobertaTokenizer, RobertaForMaskedLM

class SKEPSeq2Seq(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(
        self,
        emb_dim,
        vocab_size,
        trg_vocab_size,
        src_hidden_dim,
        trg_hidden_dim,
        attention_mode,
        batch_size,
        nlayers=2,
        nlayers_trg=2,
        dropout=0.2,
        encoder_dropout=0.2,
        decoder_dropout=0.2,
        attention_dropout=0.2,
        args=None

    ):
        """Initialize model."""
        super(SKEPSeq2Seq, self).__init__()
        self.args = args
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.src_hidden_dim = src_hidden_dim
        self.trg_hidden_dim = trg_hidden_dim
        self.attention_mode = attention_mode
        self.bidirectional = True
        self.trg_vocab_size = trg_vocab_size

        self.encoder = RobertaForMaskedLM.from_pretrained('Yaxin/roberta-large-ernie2-skep-en')
        # LSTMEncoder(emb_dim, src_hidden_dim, vocab_size, encoder_dropout=encoder_dropout)

        self.decoder = Seq2SeqDecoder(
            emb_dim,
            trg_hidden_dim,
            self.trg_vocab_size,
            batch_first=True,
            dropout=decoder_dropout,
            args=self.args
        )
        self.encoder2decoder_scr_hm = nn.Linear(50265, trg_hidden_dim, bias=False)
        self.encoder2decoder_ctx = nn.Linear(50265, trg_hidden_dim, bias=False)
        #self.encoder.requires_grad = self.args.encoder_requires_grad



    def forward(self, src, src_len, moji_id=None, moji_len=None):
        """Propogate input through the network."""
        # trg_emb = self.embedding(trg)

        src_h = self.encoder(src, src_len).logits
        cur_batch_size = src_h.size()[0]
        # src_h_m = src_h_m.view(self.encoder.num_layers, 2, cur_batch_size, self.src_hidden_dim)[-1]
        # src_h_m = torch.cat((src_h_m[0], src_h_m[1]), dim=1)
        src_h_m = torch.cat([src_h[idx][one_len - 1] for idx, one_len in enumerate(src_len)], dim=0)
        
        decoder_h_0 = torch.tanh(self.encoder2decoder_scr_hm(src_h_m))  # torch.cat((src_h_t[-1], src_h_t[-2]), 1)
        decoder_c_0 = torch.zeros((cur_batch_size, self.decoder.hidden_size)).cuda()
        #print(decoder_h_0.shape, decoder_c_0.shape)
        ctx = self.encoder2decoder_ctx(src_h)
        
        decoder_logit= self.decoder(
            (decoder_h_0, decoder_c_0),
            ctx,
            src_len
        )

        return decoder_logit

