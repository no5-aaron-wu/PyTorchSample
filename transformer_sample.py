# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : transformer_sample.py
@Project  : PyTorchSample
@Time     : 2021/12/20 14:12
@Author   : aaron-wu
@Contact_1: no5aaron@163.com
@Contact_2: wuhao@insta360.com
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2021/12/20 14:12        1.0             None
"""
import numpy
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
import time
import onnxruntime

torch.set_printoptions(precision=4, sci_mode=False)
numpy.set_printoptions(precision=4, suppress=True)

class TransformerModel(torch.nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.transformer = torch.nn.Transformer()

    def forward(self, src, tgt):
        out = self.transformer(src, tgt)
        return out


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class Generator(nn.Module):
    # 根据Decoder的隐状态输出一个词的概率分布
    # d_model是Decoder输出的大小，vocab是词典大小
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    # 全连接再加上一个softmax
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class TransformerFullModel(torch.nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(TransformerFullModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask, mem_mask):
        return self.generator(self.decode(self.encode(src, src_mask), mem_mask, tgt, tgt_mask))

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, mem_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, tgt_mask, mem_mask)


class TransformerFullModelEx(torch.nn.Module):
    def __init__(self, transformer, src_embed, tgt_embed, generator):
        super(TransformerFullModelEx, self).__init__()
        self.transformer = transformer
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask, mem_mask):
        return self.generator(self.transformer(self.src_embed(src), self.tgt_embed(tgt), src_mask, tgt_mask, mem_mask))

    def encode(self, src, src_mask):
        return self.transformer.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, mem_mask, tgt, tgt_mask):
        return self.transformer.decoder(self.tgt_embed(tgt), memory, tgt_mask, mem_mask)


def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=h, dim_feedforward=d_ff, dropout=dropout,
                                               layer_norm_eps=1e-6, batch_first=True, norm_first=True)
    encoder = nn.TransformerEncoder(encoder_layer, N, norm=layer_norm)

    decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=h, dim_feedforward=d_ff, dropout=dropout,
                                               layer_norm_eps=1e-6, batch_first=True, norm_first=True)
    decoder = nn.TransformerDecoder(decoder_layer, N, layer_norm)

    src_embed = nn.Sequential(Embeddings(d_model, src_vocab), PositionalEncoding(d_model, dropout))
    tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), PositionalEncoding(d_model, dropout))

    generator = Generator(d_model, tgt_vocab)

    model = TransformerFullModel(encoder, decoder, src_embed, tgt_embed, generator)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model


def make_model_ex(src_vocab, tgt_vocab, N=6,
                  d_model=512, d_ff=2048, h=8, dropout=0.1):
    transformer = nn.Transformer(d_model=d_model, nhead=h, num_encoder_layers=N, num_decoder_layers=N,
                                 dim_feedforward=d_ff, dropout=dropout, layer_norm_eps=1e-6, batch_first=True,
                                 norm_first=True)

    src_embed = nn.Sequential(Embeddings(d_model, src_vocab), PositionalEncoding(d_model, dropout))
    tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), PositionalEncoding(d_model, dropout))
    generator = Generator(d_model, tgt_vocab)

    model = TransformerFullModelEx(transformer, src_embed, tgt_embed, generator)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 1


def _generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0, n_head=8):
        src_len = src.size(-1)
        tgt_len = src_len if trg is None else src_len - 1

        self.src = src
        self.src_mask = torch.ones((n_head, src_len, src_len), dtype=torch.bool)
        self.src_mask = (self.src_mask & (src == pad).unsqueeze(-2).unsqueeze(-2)).view(-1, src_len, src_len)

        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.trg_mask = (self.trg_mask.unsqueeze(1) & \
                             torch.ones((n_head, tgt_len, tgt_len),
                                        dtype=torch.bool)).view(-1, tgt_len, tgt_len)
            self.ntokens = (self.trg_y != pad).data.sum()

        self.memory_mask = torch.ones((n_head, tgt_len, src_len), dtype=torch.bool)
        self.memory_mask = (self.memory_mask & (src == pad).unsqueeze(-2).unsqueeze(-2)).view(-1, tgt_len, src_len)

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


def data_gen(V, S, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        np_data = np.random.randint(1, V, size=(batch, S))
        data = torch.LongTensor(np_data)
        # data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, criterion, opt=None):
        # self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        # x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        # return loss.data[0] * norm
        return loss.data * norm


def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask, batch.memory_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


def greedy_decode(model, src, src_mask, max_len, start_symbol, n_head=8):
    # memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)

    last_prob = []
    for i in range(max_len - 1):
        tgt_mask = torch.ones((n_head, ys.size(-1), ys.size(-1)), dtype=torch.bool)
        tgt_mask = tgt_mask & Variable(subsequent_mask(ys.size(-1)).type_as(tgt_mask))
        mem_mask = Variable(torch.zeros((n_head, ys.size(-1), src.size(-1)), dtype=torch.bool))

        # out = model.decode(memory, mem_mask, Variable(ys), tgt_mask)
        # prob = model.generator(out[:, -1])
        out = model.forward(src, ys, src_mask, tgt_mask, mem_mask)
        prob = out[:, -1]
        last_prob.append(prob.detach().numpy())
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys, tgt_mask, mem_mask, last_prob


def export_onnx_test():
    src = torch.rand((10, 32, 512))
    tgt = torch.rand((20, 32, 512))

    transformer_net = TransformerModel()

    torch_output = transformer_net(src, tgt)
    print("transformer output: ", torch_output.shape)

    model_name = './export_model/TransformerModel.onnx'
    torch.onnx.export(transformer_net, (src, tgt), model_name, input_names=["src", "tgt"], output_names=["out"],
                      opset_version=11)


def full_model_test():
    # Train the simple copy task.
    V = 11
    S = 10
    n_head = 8
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    # model = make_model(V, V, N=2, h=n_head)
    model = make_model_ex(V, V, N=2, h=n_head)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    epoch_loop = 10
    for epoch in range(epoch_loop):
        model.train()
        run_epoch(data_gen(V, S, 30, 20), model, SimpleLossCompute(criterion, model_opt))
        model.eval()
        print(run_epoch(data_gen(V, S, 30, 5), model, SimpleLossCompute(criterion, None)))

    torch.save(model, './export_model/TransformerFullModel.pt')
    model.eval()
    src = Variable(torch.LongTensor([[1, 6, 7, 8, 9, 10, 5, 2, 3, 4]]))
    src_mask = Variable(torch.zeros((n_head, S, S), dtype=torch.bool))
    ys, tgt_mask, mem_mask = greedy_decode(model, src, src_mask, max_len=10, start_symbol=1)
    print(ys)

    print("export the model...")
    input_names = ["src", "tgt", "src_mask", "tgt_mask", "mem_mask"]
    output_names = ["out"]
    dynamic_axes = {'tgt': {1: 's'},
                    'tgt_mask': {1: 's', 2: 's'},
                    'mem_mask': {1: 's'}}
    # dynamic_axes = None

    model_name = './export_model/TransformerFullModel.onnx'
    torch.onnx.export(model, (src, ys[:, :-1], src_mask, tgt_mask, mem_mask), model_name, input_names=input_names,
                      output_names=output_names, dynamic_axes=dynamic_axes, opset_version=11)

    pass


def load_model_and_run():
    S = 10

    model_name = './export_model/TransformerFullModel.pt'
    model = torch.load(model_name)
    model.eval()
    n_head = model.transformer.nhead

    src = Variable(torch.LongTensor([[1, 6, 7, 8, 9, 10, 5, 2, 3, 4]]))
    src_mask = Variable(torch.zeros((n_head, S, S), dtype=torch.bool))
    ys, tgt_mask, mem_mask, last_prob = greedy_decode(model, src, src_mask, max_len=10, start_symbol=1)
    print(ys)
    for prob in last_prob:
        print(prob)


if __name__ == '__main__':
    # export_onnx_test()
    # full_model_test()
    load_model_and_run()

    pass
