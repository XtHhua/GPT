import os
import json
import regex as re
import requests
import pickle
import math
import time
from collections import defaultdict
import tiktoken

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


# 本文以tinyshakespeare数据集为例。利用tiktoken和BPE 算法进行tokenize。tokenize过程，将文本转化成一系列的数字，这些数字可以作为输入给模型。简单而言就是为模型提供了处理文本数据的基石。
data_dir = os.path.join("data", "tinyshakespeare")
input_file_path = os.path.join(data_dir, "input.txt")
if not os.path.exists(input_file_path):
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    os.makedirs(data_dir)
    with open(input_file_path, "w") as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, "r") as f:
    data = f.read()
n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(data_dir, "train.bin"))
val_ids.tofile(os.path.join(data_dir, "val.bin"))


# 模型超参数的设置
class GPTConfig:
    def __init__(self, vocab_size, **kwargs):
        self.vocab_size = vocab_size
        for key, value in kwargs.items():
            setattr(self, key, value)


class CustomConfig(GPTConfig):
    n_layer = 8
    n_head = 8
    n_embd = 256
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    dropout = 0.1
    compile = True
    device = "cuda"
    num_workers = 0
    max_iters = 2e4
    batch_size = 4
    block_size = 64
    learning_rate = 6e-4
    betas = (0.9, 0.95)
    weight_decay = 1e-1
    grad_norm_clip = 1.0


vocab_size = len(train_ids)
config = CustomConfig(vocab_size=vocab_size)


# Dataloaders定义
# 导入先前保存的.bin文件
data_dir = os.path.join("data", "tinyshakespeare")
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")


class ShakespeareDataset(Dataset):
    def __init__(self, split, block_size=128, device_type="cuda"):
        assert split in {"train", "test"}
        self.split = split
        self.block_size = block_size
        self.device_type = device_type
        self.data = train_data if split == "train" else val_data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # x y 取相同block size长度的切片，但是，y比x往后多走了一个
        x = torch.from_numpy(self.data[idx : idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1 : idx + 1 + self.block_size].astype(np.int64))

        if self.device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to("cuda", non_blocking=True), y.pin_memory().to("cuda", non_blocking=True)
        else:
            x, y = x.to("cpu"), y.to("cpu")
        return x, y


# create dataset and dataloader
train_dataset = ShakespeareDataset("train", config.block_size, config.device)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=False)
test_dataset = ShakespeareDataset("test", config.block_size, config.device)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=False)


# GELU Activation Function
# GELU（高斯误差线性单元）激活函数是一种非线性激活函数，于2016年由Hendrycks和Gimpel引入。它是ReLU激活函数的平滑近似，并且在某些深度学习模型中表现比ReLU函数更好。GELU函数具有几个理想的特性，例如可微性和范围从-1到正无穷。研究表明，GELU函数可以提高深度学习模型的训练速度和准确性，特别是在自然语言处理任务中。
class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


# Causal Self Attention
# 因果（causal）自注意力是Transformer架构中使用的自注意力机制的一个变种，它是GPT模型的关键组件之一。两者之间的区别在于，因果自注意力将注意力机制限制在仅查看序列中先前的标记，从而适用于生成文本。
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.dropout = config.dropout
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        # batch_size, seq_len, emb_dim
        B, T, C = x.size()

        # (b, seq_len, emb_dim) --> (b, seq_len, emb_dim * 3) --> (b, seq_len, emb_dim)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (b, h, seq_len, d_k)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (b, h, seq_len, d_k)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (b, h, seq_len, d_k)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            # (b, h, seq_len, d_k) matmul (b, h, d_k, seq_len) --> (b, h, seq_len, seq_len)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # diagonal mask
            # fill 0 mask with super small number so it wont affect the softmax weight
            # (batch_size, h, seq_len, seq_len)
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)

            # (b, h, seq_len, seq_len) matmul (b, h, seq_len, d_k) --> (b, h, seq_len, d_k)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


# Decoder Block
class Block(nn.Module):
    """GPT decoder block"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)

        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(config.n_embd, 4 * config.n_embd),
                act=NewGELU(),
                c_proj=nn.Linear(4 * config.n_embd, config.n_embd),
                dropout=nn.Dropout(config.resid_pdrop),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x):

        # (batch_size, seq_len, emb_dim)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


# GPT 模型
class GPT(nn.Module):
    """GPT Language Model"""

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        self.transformer = nn.ModuleDict(
            dict(
                ### nn.Embedding 类似于一个查找，通过oneshot去查找对应的token的向量表示
                # 和nn.linear不同的是，她俩的权重是转置关系
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                # 现在呢，也流行用一个可学习参数，来表示位置编码，再通过embedding的包，来查找固定的位置编码
                # 如果不懂，可以去看看上面那个medium的博客
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.embd_pdrop),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None):
        # 模型在forward阶段，会根据输入的sequence长度，去计算embedding，因此在输入长度会影响embedding的内存大小
        # 然而在self-attention中，k、q、v的计算，也是需要sequence的长度的
        # 所以输入的sequence越长，在推理的时候，所需要的GPU显存也就越高
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        # positional token, shape (1, t)
        # 生成一个1-t的整数
        # tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, t]])
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        # (b, t, n_embd) -- > # (b, t, vocab_size)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        # -1 at output will be ignored
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b, t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)  # 其实是在__call__中默认调用了forward，就是默认是推理啦
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                # torch.multinomial函数是PyTorch中用于从给定的概率分布中随机抽取样本的函数。
                # 这个函数对于实现基于概率的抽样非常有用，
                # 特别是在处理诸如文本生成这类需要根据预测概率随机选择下一个输出的任务中。
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                # torch.topk Returns the k largest elements of the given input tensor along a given dimension.
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# 训练代码
class Trainer:
    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)
        self.device = config.device
        self.model = self.model.to(self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            # pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss = model(x, y)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks("on_batch_end")
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break


model = GPT(config).to(config.device)
if config.compile:
    model = torch.compile(model)
trainer = Trainer(config, model, train_dataset)


def batch_end_callback(trainer):
    if trainer.iter_num % 500 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")


trainer.set_callback("on_batch_end", batch_end_callback)
trainer.run()

# 生成文本
text = "Lord:\nRise! My people, conquer the north!"
sample_ids = torch.Tensor(enc.encode_ordinary(text)).long()
sample_ids = torch.unsqueeze(sample_ids, 0).to(config.device)
result = model.generate(sample_ids, max_new_tokens=50, temperature=1, do_sample=False, top_k=None)
print(enc.decode(result.detach().cpu().tolist()[0]))
