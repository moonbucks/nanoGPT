# original file: https://github.com/karpathy/nanoGPT
# updates # Copyright (c) Meta Platforms, Inc. and affiliates
"""
This training script updates NanoGPT to run with either FSDP, or FSDP + Tensor Parallel (2D).
Usage:
a - ensure run_tp.sh has the right amount of gpus (_nproc_per_node = your gpu count)
b - you can control FSDP vs FSDP + TP using config/config_2D.py, setting the 
    use_tensor_parallel: bool = True/False
c - to run, bash run_tp.sh

"""

import inspect
import math
import os
import pickle
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from model import GPT, GPTConfig

from torch.distributed._tensor import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PairwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from torch.nn import functional as F

from gpu_memory import Memory_Maximizer

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = "out"
eval_interval = 2000
log_interval = 2
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
train_iters = 200000
# data
#dataset = "openwebtext"
dataset = "shakespeare_char"
gradient_accumulation_steps = 1  # used to simulate larger batch sizes
batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 4e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
backend = "nccl"  # 'nccl', 'gloo', etc.

compile = False  # use PyTorch 2.0 to compile the model to be faster

# Init TP
_multi_gpu = int(os.environ.get("RANK", -1)) != -1  # verify distributed run

assert _multi_gpu, "this config assumes distributed setup - multi-gpu not ready here."


_rank = int(os.environ["RANK"])
_local_rank = int(os.environ["LOCAL_RANK"])

world_size = int(os.environ["WORLD_SIZE"])  # total number of training processes
device = f"cuda:{_local_rank}"
torch.cuda.set_device(device)
master_process = _rank == 0  # this process will do logging, checkpointing etc.
#seed_offset = _rank  # each process gets a different seed

dist.init_process_group(backend=backend, rank=_rank, world_size=world_size)

# wrapper to avoid cluttering with if rank==0...
def rank_print(x):
    if _rank == 0:
        print(x)

_gpu_mem_tracker = Memory_Maximizer(rank=_rank)

tensor_parallel_size = 2

twod_mesh = DeviceMesh(
    device_type="cuda",
    mesh=torch.arange(0, world_size).view(-1, tensor_parallel_size),
)

rank_print(f"{twod_mesh=}\n")


if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
torch.backends.cuda.enable_mem_efficient_sdp(enabled=False)
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast

# poor man's data loader
data_dir = os.path.join('data', dataset)
rank_print(f"{data_dir=}")
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line

# init a new model from scratch
rank_print("Initializing a new model from scratch")

tp_device_mesh = DeviceMesh(device_type, list(range(world_size)))
rank_print(f"{tp_device_mesh=}")
rank_print(f"tp_size = {tp_device_mesh.size(0)}")

if meta_vocab_size is None:
    print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
#model_args['mesh'] = tp_device_mesh
gptconf = GPTConfig(**model_args)
model = GPT(tp_device_mesh, gptconf)
#model = GPT(gptconf)

model.to(device)

# we need this or else calcing mfu in fsdp = sharded model size...
_mfu_model_params = model.get_num_params()
_current_model_params = _mfu_model_params / 1e6

def tp(model, n_layer, mesh):
  #for name, modules in model.named_modules():
  #  print(name, type(modules))

  for i in range(n_layer):
    block = model.get_submodule(f'transformer.h.{i}')
    parallelize_module(block, mesh, {'attn.q': ColwiseParallel(),
                                     'attn.k': ColwiseParallel(),
                                     'attn.v': ColwiseParallel(),
                                     'attn.c_proj': RowwiseParallel(),
                                     #'mlp.c_fc': ColwiseParallel(),
                                     #'mlp.c_proj': RowwiseParallel()})
                                     'mlp': PairwiseParallel()}) # why not work?

    #if os.getenv('RANK') == '0' and i == 0:
    #  print('layer', i)
    #  print(block.mlp.c_fc.weight)
    #  print(block.mlp.c_proj.weight)
  return model


pp_device_mesh = DeviceMesh(device_type, list(range(world_size)))
pp_chunks = 4
pp_groups = pp_device_mesh.get_dim_groups()[0]

class LossWrapper(nn.Module):
  def __init__(self, module, loss_fn):
    super().__init__()
    self.module = module
    self.loss_fn = loss_fn

class TrivialLossWrapper(LossWrapper):
  def forward(self, x, targets=None):
    logits, _loss = self.module(x, targets) # Can we directly return _loss?
    #return _loss
    return self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)


def pp(model):
  from pippy.IR import annotate_split_points, PipeSplitWrapper 
  from pippy import split_into_equal_size
  from pippy.compile import compile_stage

  #annotate_split_points(model, {})
  split_policy = split_into_equal_size(world_size)

  X, Y = get_batch('train')
  input_names = ['idx', 'targets']
  sig = inspect.signature(model.forward)
  print('sig', sig)

  concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}
  #concrete_args['idx'] = X.to(f'cuda:{_rank}') 
  concrete_args['targets'] = Y.to(X.device) 
  b, t = X.size()
  concrete_args['b'] = b
  concrete_args['t'] = t
  concrete_args['device'] = X.device 
  #concrete_args['x'] = X.to(f'cuda:{_rank}') 
  #concrete_args['loss_fn'] = F.cross_entropy
  print('concrete_args', concrete_args)

  model_with_loss = TrivialLossWrapper(model, F.cross_entropy)
  stage = compile_stage(model, _rank, world_size, pp_chunks, device, pp_groups, 
  #stage = compile_stage(model_with_loss, _rank, world_size, pp_chunks, device, pp_groups, 
            example_inputs=[X], split_policy=split_policy,
            concrete_args=concrete_args)
  
  print(stage)
  return model, stage 

def pp_fixed(model):
  from pippy.IR import annotate_split_points, PipeSplitWrapper 
  from pippy import split_into_equal_size
  from pippy.compile import compile_stage
  from pippy.microbatch import TensorChunkSpec, sum_reducer

  #annotate_split_points(model, {})
  #split_policy = split_into_equal_size(world_size)

  X, Y = get_batch('train')
  input_names = ['idx', 'targets']
  sig = inspect.signature(model.forward)
  rank_print('sig: {sig}')

  concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}
  #concrete_args['idx'] = X.to(f'cuda:{_rank}')
  #concrete_args['targets'] = Y.to(X.device) 
  #concrete_args['bbatch'] = 12
  #concrete_args['t'] = 1024
  #concrete_args['device'] = X.device 

  #for wrapper
  #concrete_args['x'] = X.to(f'cuda:{_rank}') 
  #concrete_args['loss_fn'] = F.cross_entropy
  rank_print(f'concrete_args: {concrete_args}')

  model_with_loss = TrivialLossWrapper(model, F.cross_entropy)
  #stage = compile_stage(model, _rank, world_size, pp_chunks, device, pp_groups, 
  output_chunk_spec = (TensorChunkSpec(0), sum_reducer)
  stage = compile_stage(model, _rank, world_size, pp_chunks, device, pp_groups, 
            example_inputs=[X, Y], 
            #concrete_args=concrete_args,
            output_chunk_spec=output_chunk_spec)
  
  print(f'[Rank{_rank}] {stage.submod.print_readable()}')
  return model, stage 

#model = tp(model, n_layer, tp_device_mesh)
model, stage = pp_fixed(model)

rank_print(stage)

# optimizer
# new PyTorch nightly has a new 'fused' option for AdamW that is much faster

use_fused = (device_type == "cuda") and (
    "fused" in inspect.signature(torch.optim.AdamW).parameters
)

rank_print(f"Optimizer = using fused AdamW: {use_fused}")
extra_args = dict(fused=True) if use_fused else dict()
#optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, **extra_args)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            print("after get batch ", k)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# training loop
X, Y = get_batch("train")  # fetch the very first batch
local_iter_num = 0  # number of iterations in the lifetime of this process
running_mfu = -1.0
eval_interval = 1
warmup = 5
iter_time_accumulator = 0.0
iter_count = 0

_tp_size = world_size

# num_params = model.get_num_params()
# print(f"{num_params=}")

_gpu_mem_tracker.start()

def pp_train(stage):
  global iter_count
  train_iters = 10
  optimizer = torch.optim.AdamW(stage.submod.parameters(), lr=learning_rate)
  local_iter_num = 0
  while local_iter_num < train_iters:
      optimizer.zero_grad()
      t0 = time.perf_counter()
      X, Y= get_batch('train')
      if _rank == 0:
        #X, Y = get_batch("train") # we are not actually doing training
        out = stage(X)
      elif _rank == 3:
        out = stage(Y)
      else :
        out = stage()

      #torch.distributed.barrier()
      optimizer.step()

      # timing and logging
      t1 = time.perf_counter()
      dt = t1 - t0
      _gpu_mem_tracker.update()
      local_iter_num += 1
      iter_count += 1
      print(f'L394 iter{local_iter_num} finished at rank {_rank}')
  
def tp_train():
  while local_iter_num < train_iters:
      lr = get_lr(local_iter_num) if decay_lr else learning_rate
      for param_group in optimizer.param_groups:
        param_group['lr'] = lr

      t0 = time.perf_counter()
      logits, loss = model(X, Y)
      # immediately async prefetch next batch while model is doing the forward pass on the GPU
      X, Y = get_batch("train")
      loss.backward()
      optimizer.step()
      optimizer.zero_grad(set_to_none=True)
      torch.distributed.barrier()

      # timing and logging
      t1 = time.perf_counter()
      dt = t1 - t0
      _gpu_mem_tracker.update()

      lossf = loss.item()  # loss as float. note: this is a CPU-GPU sync point

      if iter_num >= warmup:
          # if local_iter_num >= 3:  # let the training loop settle a bit

          mfu = model.estimate_mfu(
              _mfu_model_params,
              batch_size,  # / _fsdp_size * world_size,  #  * gradient_accumulation_steps,
              dt,
              #config_file=gptconf,
              #tp_size=_tp_size,
          )
          running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
          rank_print(
              f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
          )
          iter_time_accumulator += dt
          iter_count += 1
      else:
          rank_print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

      iter_num += 1
      local_iter_num += 1
      # rank_print(f"iter {iter_num} completed...")

      # termination conditions
      if iter_num > train_iters:
          break


pp_train(stage)
#dist.barrier()
rank_print(
    f"\nTraining completed.\n"
)

# display run stats
gpu_type = torch.cuda.get_device_name(0)
gpu_count = dist.get_world_size()
# model_params = model.get_num_params() # /1e6
rank_print(f"\n----- Performance Stats --------\n")

if _rank == 0:
    _gpu_mem_tracker.stop()
rank_print(f"\nModel Size:  {_current_model_params:.2f}M")
rank_print(f"Run completed with {gpu_count} gpus, of type {gpu_type}")
rank_print(f"Running MFU final = {running_mfu*100:.2f}%")
iter_avg = round(iter_time_accumulator / iter_count, 4)
rank_print(
    f"Avg iter speed (in seconds): {iter_avg}, with {iter_count} iterations averaged.\n"
)
dist.destroy_process_group()
