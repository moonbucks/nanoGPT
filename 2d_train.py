# original file: https://github.com/karpathy/nanoGPT
# updates # Copyright (c) Meta Platforms, Inc. and affiliates
"""
This training script updates NanoGPT to run with either TP, PP, or TP+PP (2D).
Usage:
a - ensure run_tp.sh has the right amount of gpus (_nproc_per_node = your gpu count)
b - gpurun4 torchrun --nproc-per-node 4 2d_train.py 
"""

import argparse
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

def get_args():
  # default config values designed to train a gpt2 (124M) on OpenWebText

  def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't', '1'): 
        return True
    elif v.lower() in ('false', 'f', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean expected.')

  # I/O
  parser = argparse.ArgumentParser()
  parser.add_argument('--out_dir', type=str, default='out')
  parser.add_argument('--eval_interval', type=int, default=2000)
  parser.add_argument('--log_interval', type=int, default=2)
  parser.add_argument('--eval_iters', type=int, default=200)
  parser.add_argument('--eval_only', type=str_to_bool, default=False) # if True, script exits right after the first eval
  parser.add_argument('--always_save_checkpoint', type=str_to_bool, default=True)  # if True, always save a checkpoint after each eval
  parser.add_argument('--init_from', type=str, default="scratch")  # 'scratch', 'resume', 'gpt2*'
  parser.add_argument('--train_iters', type=int, default=200000)
  parser.add_argument('--seed', type=int, default=1337)

  # data
  parser.add_argument('--dataset', type=str, default="shakespeare_char") # "openwebtext"
  parser.add_argument('--gradient_accumulation_steps', type=int, default=1) # used to simulate larger batch sizes
  parser.add_argument('--batch_size', type=int, default=12)  # if gradient_accumulation_steps > 1, this is the micro-batch size
  parser.add_argument('--block_size', type=int, default=1024)

  # model
  parser.add_argument('--n_layer', type=int, default=12)
  parser.add_argument('--n_head', type=int, default=12)
  parser.add_argument('--n_embd', type=int, default=768)
  parser.add_argument('--dropout', type=float, default=0.0)  # for pretraining 0 is good, for finetuning try 0.1+
  parser.add_argument('--bias', type=str_to_bool, default=False) 

  # adamw optimizer
  parser.add_argument('--learning_rate', type=float, default=4e-4)  # max learning rate
  parser.add_argument('--max_iters', type=int, default= 600000)  # total number of training iterations
  parser.add_argument('--weight_decay', type=float, default=1e-2)
  parser.add_argument('--beta1', type=float, default=0.9)
  parser.add_argument('--beta2', type=float, default=0.95)
  parser.add_argument('--grad_clip', type=float, default=1.0) # clip gradients at this value, or disable if == 0.0
  parser.add_argument('--decay_lr', type=str_to_bool, default=True) # whether to decay the learning rate
  parser.add_argument('--warmup_iters', type=int, default=2000)  
  parser.add_argument('--lr_decay_iters', type=int, default=600000) 
  parser.add_argument('--min_lr', type=float, default= 6e-5)  # minimum learning rate 
  
  # distributed
  parser.add_argument('--backend', type=str, default="nccl")  # 'nccl', 'gloo', etc.
  parser.add_argument('--compile', type=str_to_bool, default=False) # use PyTorch 2.0 to compile the model to be faster
  parser.add_argument('--rank', type=int, default=int(os.environ["RANK"]))
  parser.add_argument('--local_rank', type=int, default=int(os.environ["LOCAL_RANK"]))
  parser.add_argument('--world_size', type=int, default=int(os.environ["WORLD_SIZE"]))
  parser.add_argument('--device', type=str, default=f"cuda:{os.environ['LOCAL_RANK']}")
  parser.add_argument('--master_process', type=str_to_bool, default=bool(os.environ['RANK']==0)) 
  parser.add_argument('--tp_size', type=int, default=2)
  parser.add_argument('--pp_size', type=int, default=2)

  parser.add_argument('--debug', dest='debug', action='store_true')

  args = parser.parse_args()

  return args


def rank_print(x):
    _rank = os.getenv('RANK')
    if _rank == 0:
        print(x)

def get_batch(data, args):
    ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + args.block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + args.block_size]).astype(np.int64))
            for i in ix
        ]
    )
    device_type = "cuda" if "cuda" in args.device else "cpu"  
    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(args.device, non_blocking=True), y.pin_memory().to(
            args.device, non_blocking=True
        )
    else:
        x, y = x.to(args.device), y.to(args.device)
    return x, y


def tp_attention(model, name, mesh, tp_dim=0, q='q', k='k', v='v', o='c_proj'):
  layer = model.get_submodule(name)
  parallelize_module(layer, mesh, { q: ColwiseParallel(),
                                    k: ColwiseParallel(),
                                    v: ColwiseParallel(),
                                    o: RowwiseParallel() },
                                    tp_mesh_dim=tp_dim)

  return model

def tp_mlp(model, name, mesh, tp_dim=0, mlp='mlp'):
  layer = model.get_submodule(name)
  parallelize_module(layer, mesh, { mlp: PairwiseParallel() }, tp_mesh_dim=tp_dim)

  return model


def tp(model, n_layer, mesh, offset=0, tp_dim=0):
  for i in range(n_layer):
    block = model.get_submodule(f'transformer.h.{i + offset}')
    parallelize_module(block, mesh, {'attn.q': ColwiseParallel(),
                                     'attn.k': ColwiseParallel(),
                                     'attn.v': ColwiseParallel(),
                                     'attn.c_proj': RowwiseParallel(),
                                     'mlp': PairwiseParallel()},
                                     tp_mesh_dim=tp_dim) 

  return model

def pp(model, pp_device_mesh, args):
  from pippy.IR import annotate_split_points, PipeSplitWrapper 
  from pippy import split_into_equal_size
  from pippy.compile import compile_stage
  from pippy.microbatch import TensorChunkSpec, sum_reducer

  #annotate_split_points(model, {})
  # for now we are not going to use split_policy, concrete_args

  pp_chunks = args.world_size 
  pp_groups = pp_device_mesh.get_dim_groups()[0]

  output_chunk_spec = (TensorChunkSpec(0), sum_reducer)
  stage = compile_stage(model, args.rank, args.world_size, pp_chunks, pp_device_mesh, pp_groups, 
            example_inputs=[X, Y], 
            output_chunk_spec=output_chunk_spec)
  
  print(f'[Rank{_rank}] {stage.submod.print_readable()}')
  return model, stage 

def pp_and_tp(model, mesh, args, train_data):
  from pippy.compile import compile_stage
  from pippy.microbatch import TensorChunkSpec, sum_reducer

  pp_dim, tp_dim = 0, 1
  pp_rank, tp_rank = args.local_rank // args.tp_size, args.local_rank % args.tp_size
  pp_groups = mesh.get_dim_groups()[pp_dim]

  # TP
  tp(model, args.n_layer, mesh, 0, tp_dim)

  X, Y = get_batch(train_data, args)

  # PP
  stage = compile_stage(model, pp_rank, args.world_size, args.pp_size, args.device, pp_groups, 
            example_inputs=[X, Y],
            )
  
  return model, stage

def even_cut(model, args, pp_size, cut={}):
  from pippy.IR import annotate_split_points, PipeSplitWrapper 
  cutpoint = args.n_layer // pp_size
  for i in range(args.n_layer):
    name = f'transformer.h.{i}'
    if i > 0 and i % cutpoint == 0:
      cut[name] = PipeSplitWrapper.SplitPoint.BEGINNING # or END

  annotate_split_points(model, cut)

def pp_and_tp_fg(model, mesh, args, train_data, tp_attn_layers=None, tp_mlp_layers=None, cut_fn=even_cut):
  from pippy.compile import compile_stage
  from pippy.microbatch import TensorChunkSpec, sum_reducer

  pp_dim, tp_dim = 0, 1
  pp_rank, tp_rank = args.local_rank // args.tp_size, args.local_rank % args.tp_size
  pp_groups = mesh.get_dim_groups()[pp_dim]

  # TP
  # Apply TP to layers if layer_id is in tp_attn / tp_mlp 
  tp_attn_layers = list(range(args.n_layer)) if tp_attn_layers is None else tp_attn_layers
  tp_mlp_layers = list(range(args.n_layer)) if tp_mlp_layers is None else tp_mlp_layers
  for i in range(args.n_layer):
    name = f'transformer.h.{i}'
    att = tp_attention(model, f'{name}.attn', mesh, tp_dim)
    mlp = tp_mlp(model, f'{name}', mesh, tp_dim)

  X, Y = get_batch(train_data, args)

  if args.local_rank == 0:
    for name, module in model.named_modules():
      print(name)

  # PP
  cut_fn(model, args, args.pp_size)
  stage = compile_stage(model, pp_rank, args.world_size, args.pp_size, args.device, pp_groups, 
            example_inputs=[X, Y],
            )

  if args.local_rank == 0:
    print(stage.submod)

  return model, stage

# learning rate decay scheduler (cosine with warmup)
def get_lr(it, args):
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return args.learning_rate * it / args.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > args.lr_decay_iters:
        return args.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)

def pp_tp_train(stage, mesh, args, train_data):
  iter_count = 0
  pp_dim, tp_dim = 0, 1
  pp_rank, tp_rank = args.local_rank // args.tp_size, args.local_rank % args.tp_size
  pp_groups = mesh.get_dim_groups()[pp_dim]

  train_iters = 10 if args.debug else args.train_iters
  optimizer = torch.optim.AdamW(stage.submod.parameters(), lr=args.learning_rate)
  local_iter_num = 0
  while local_iter_num < train_iters:
      optimizer.zero_grad()
      t0 = time.perf_counter()
      X, Y = get_batch(train_data, args)
      if pp_rank == 0:
        out = stage(X)
      elif pp_rank == args.pp_size - 1:
        out = stage(Y)
      else :
        out = stage()
      optimizer.step()
      t1 = time.perf_counter()
      dt = t1 - t0
      _gpu_mem_tracker.update()
      local_iter_num += 1
      iter_count += 1

  return iter_count

def pp_train(stage, args, train_data):
  iter_count = 0

  train_iters = 10 if args.debug else args.train_iters
  optimizer = torch.optim.AdamW(stage.submod.parameters(), lr=args.learning_rate)
  local_iter_num = 0
  while local_iter_num < train_iters:
      optimizer.zero_grad()
      t0 = time.perf_counter()
      X, Y = get_batch(train_data, args)
      if args.rank == 0:
        out = stage(X)
      elif args.rank == args.world_size - 1:
        out = stage(Y)
      else :
        out = stage()
      optimizer.step()
      t1 = time.perf_counter()
      dt = t1 - t0
      _gpu_mem_tracker.update()
      local_iter_num += 1
      iter_count += 1

  return iter_count
  
def tp_train():
  iter_count = 0
  local_iter_num = 0
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
  while local_iter_num < train_iters:
      lr = get_lr(local_iter_num) if decay_lr else learning_rate
      for param_group in optimizer.param_groups:
        param_group['lr'] = lr

      t0 = time.perf_counter()
      logits, loss = model(X, Y)
      # immediately async prefetch next batch while model is doing the forward pass on the GPU
      X, Y = get_batch(train_data, batch_size, block_size, device)
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
              batch_size,
              dt,
              #config_file=gptconf,
              #tp_size=tp_size,
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

      # termination conditions
      if iter_num > train_iters:
          break

  return iter_count

if __name__ == '__main__':

  _multi_gpu = int(os.environ.get("RANK", -1)) != -1  # verify distributed run
  assert _multi_gpu, "this config assumes distributed setup - multi-gpu not ready here."

  args = get_args()

  device_type = "cuda" if "cuda" in args.device else "cpu"  # for later use in torch.autocast
  torch.cuda.set_device(args.device)
  
  _gpu_mem_tracker = Memory_Maximizer(rank=args.rank)

  dist.init_process_group(backend=args.backend, rank=args.rank, world_size=args.world_size)

  if args.master_process:
      os.makedirs(args.out_dir, exist_ok=True)

  torch.manual_seed(args.seed)
  torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
  torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
  torch.backends.cuda.enable_mem_efficient_sdp(enabled=False)

  # poor man's data loader
  data_dir = os.path.join('data', args.dataset)
  train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
  val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

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
  model_args = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, 
                    block_size=args.block_size, bias=args.bias, vocab_size=None, 
                    dropout=args.dropout) # start with model_args from command line

  # init a new model from scratch
  rank_print("Initializing a new model from scratch")

  oned_mesh = DeviceMesh(device_type, list(range(args.world_size)))
  twod_mesh = DeviceMesh(
      device_type=device_type,
      mesh=torch.arange(0, args.world_size).view(-1, args.tp_size),
  )

  if meta_vocab_size is None:
      print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
  model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304

  gptconf = GPTConfig(**model_args)
  model = GPT(twod_mesh, gptconf, args.device, args.pp_size)
  model.to(args.device)

  _mfu_model_params = model.get_num_params()
  _current_model_params = _mfu_model_params / 1e6


  #model = tp(model, args.n_layer, oned_mesh)
  #model, stage = pp(model, oned_mesh, args)
  #model, stage = pp_and_tp(model, twod_mesh, args, train_data)
  model, stage = pp_and_tp_fg(model, twod_mesh, args, train_data)


  # optimizer
  # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
  use_fused = (device_type == "cuda") and (
      "fused" in inspect.signature(torch.optim.AdamW).parameters
  )

  # training loop
  running_mfu = -1.0
  eval_interval = 1
  warmup = 5
  iter_time_accumulator = 0.0

  _gpu_mem_tracker.start()

  iter_count = pp_tp_train(stage, twod_mesh, args, train_data)
  #iter_count = pp_train(stage, args, train_data)

  if args.master_process:
      _gpu_mem_tracker.stop()

  # display run stats
  rank_print(f"\nTraining completed.\n")

  gpu_type = torch.cuda.get_device_name(0)
  gpu_count = dist.get_world_size()
  rank_print(f"\n----- Performance Stats --------\n")
  rank_print(f"\nModel Size:  {_current_model_params:.2f}M")
  rank_print(f"Run completed with {gpu_count} gpus, of type {gpu_type}")
  rank_print(f"Running MFU final = {running_mfu*100:.2f}%")
  iter_avg = round(iter_time_accumulator / iter_count, 4)
  rank_print(
      f"Avg iter speed (in seconds): {iter_avg}, with {iter_count} iterations averaged.\n"
  )

  dist.destroy_process_group()
