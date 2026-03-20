# SimpleSim

A cycle-level GPU performance simulator for attention kernels, implementing the
"Feeds and Speeds" bottleneck methodology from the FlashAttention-4 paper.

## Quick Start

```bash
pip install pyyaml tabulate
python -m examples.flash_attention
```

## What it does

Given a **hardware model** (throughput of each compute and memory unit) and an
**algorithm workload** (how many ops each unit needs to retire), SimpleSim
computes the minimum cycle count for every unit and identifies the bottleneck.

The key improvement over a plain Roofline model is precise modelling of **shared
memory operand re-reads** caused by MMA tiling: when the output tile exceeds the
hardware MMA tile size, operands are re-read multiple times.  This is the
dominant effect in FlashAttention-4 on Blackwell.

## Validated results

The simulator exactly reproduces FlashAttention-4 Table 1 (B200 per-SM cycles):

| Config | MMA | SMEM | SFU | Bottleneck |
|--------|-----|------|-----|------------|
| M=N=d=128 | 1024 | 768 | 1024 | MMA + SFU (tied) |
| M=256, N=d=128 | 2048 | 1536 | 2048 | MMA + SFU (tied) |

## Usage

```python
from simplesim import load_gpu_config, MMAOp, TiledWorkload, CycleSimulator
from simplesim.analysis import print_report

hw = load_gpu_config("configs/b200.yaml")

wl = TiledWorkload(
    name="fa4_fwd_M128",
    mma_ops=[
        MMAOp("QKt", output_shape=(128, 128), reduction_dim=128,
              operand_a_source="smem", operand_b_source="smem", dtype_bytes=2),
        MMAOp("PV",  output_shape=(128, 128), reduction_dim=128,
              operand_a_source="tmem", operand_b_source="smem", dtype_bytes=2),
    ],
    elementwise_ops={"sfu": 128*128, "cuda_core": 4*128*128},
    hbm_bytes=(128+128+128+128)*128*2,
)

result = CycleSimulator(hw).simulate_tiled(wl)
print_report(result)
```

See `ARCHITECTURE.md` for a full component guide, the MMAOp formula derivation,
and instructions for adding new GPUs and workloads.
