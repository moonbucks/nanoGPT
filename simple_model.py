import argparse
import os
import time

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.profiler import profile, record_function, ProfilerActivity

class MLPModule(torch.nn.Module):
    def __init__(self, d_hid):
        super(MLPModule, self).__init__()
        self.net1 = torch.nn.Linear(d_hid, d_hid, bias=False, dtype=torch.bfloat16)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        x = x.to(torch.bfloat16)
        x = self.net1(x)
        x = x.to(torch.bfloat16)
        x = self.net2(x)
        return x


class ExampleCode(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mlps = nn.ModuleDict(
            dict(
                m=nn.ModuleList(
                    [MLPModule(args.hidden_size) for _ in range(args.n_layer)]
                ),
            )
        )
        self.mse_loss = torch.nn.MSELoss(reduction="sum")

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        use_fused = False
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    def forward(self, x, y=None):
        for block in self.mlps.m:
            x = x.to(torch.bfloat16)
            x = block(x)

        if y is not None:
            x = x.to(torch.bfloat16)
            y = y.to(torch.bfloat16)
            loss = self.mse_loss(x, y)
            loss = loss.to(torch.bfloat16)
            # loss = None
        else:
            loss = None

        return x, loss

def get_args_mlp():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=1024)

    # distributed
    parser.add_argument(
        "--backend", type=str, default="nccl"
    )  # 'nccl', 'gloo', etc.
    parser.add_argument(
        "--global_rank", type=int, default=int(os.environ["RANK"])
    )
    parser.add_argument(
        "--rank", type=int, default=int(os.environ["LOCAL_RANK"])
    )
    parser.add_argument(
        "--world_size", type=int, default=int(os.environ["WORLD_SIZE"])
    )
    parser.add_argument(
        "--device", type=str, default=f"cuda:{os.environ['LOCAL_RANK']}"
    )

    parser.add_argument("--train_iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=10)

    args = parser.parse_args()

    return args

def get_rand(args, specific_device=None):
    device = specific_device if specific_device is not None else args.device
    x = torch.rand(
        (args.batch_size, args.hidden_size),
        device=device,
        dtype=torch.float16,
    )
    y = torch.rand(
        (args.batch_size, args.hidden_size),
        device=device,
        dtype=torch.float16,
    )
    return x, y

if __name__ == "__main__":
    _multi_gpu = int(os.environ.get("RANK", -1)) != -1  # verify distributed run
    assert (
        _multi_gpu
    ), "this config assumes distributed setup - multi-gpu not ready here."

    args = get_args_mlp()

    device_type = (
        "cuda" if "cuda" in args.device else "cpu"
    )  # for later use in torch.autocast
    torch.cuda.set_device(args.device)

    dist.init_process_group(
        backend=args.backend,
        rank=args.global_rank,
        world_size=args.world_size,
        init_method="env://",
    )

    torch.manual_seed(args.seed+args.global_rank) # args.rank is offset for ddp

    model = ExampleCode(args)
    model.to(args.device)
    optimizer = model.configure_optimizers(1e-1, 6e-5, (0.9, 0.95), 'cuda')

    model = DDP(model, device_ids=[args.rank])

    # 2 for forward, 4 for backward
    flops_per_iter = 2 * args.hidden_size * args.hidden_size * 2 * args.n_layer * args.batch_size 
    flops_promised = 118.5e12 # A100 BF16 - 312e12, TF32 - 156e12 FP32 - 19.5e12
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
        schedule=torch.profiler.schedule(
            skip_first=10, wait=0, warmup=20, active=10, repeat=1
        ), 
    ) as prof:
        for i in range(args.warmup_iters + args.train_iters):
            torch.cuda.reset_peak_memory_stats()
            X, Y = get_rand(args)
            t0 = time.time()
            logits, loss = model(X, Y)
            #loss.backward()
            #optimizer.step()
            #optimizer.zero_grad(set_to_none=True) 
            t1 = time.time()
            dt = t1-t0
            flops_achieved = flops_per_iter / dt
            mfu = flops_achieved / flops_promised
            prof.step()
            if i > args.warmup_iters:
                print(f"iter {i}: dt: {dt} mfu {mfu*100:.2f}% peak memory: {torch.cuda.max_memory_allocated(device=args.device)/1024/1024/1024:.4f}GB")
    prof.export_chrome_trace(f"traces.json")

    destroy_process_group()
