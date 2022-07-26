
# Introduction

This is an example of using the Slurm cluster to speed up the Torch model training.


# Slurm Cluster Usage

usage: args.type = 0 for single gpu, 1 for torch.distributed.launch, 2 for slurm.

Here is an example on V100:
``` python


# torch.distributed.launch
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=22222 ddp.py --epoch=100 --type=1 --data_dir=data/

# eval
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=22222 ddp.py --type=1 --data_dir=data/ --eval=1

# slurm, 4 gpu on 1 node

srun --mpi=pmi2 -p stc1_v100_32g -n 4 --gres=gpu:4 --ntasks-per-node=4  --job-name=test --kill-on-bad-exit=1 python slurm_test.py --epoch=200 --type=2

# slurm, 16 gpu on two nodes (8 V100 for each node)

srun --mpi=pmi2 -p stc1_v100_32g -n 16 --gres=gpu:8 --ntasks-per-node=8  --job-name=test --kill-on-bad-exit=1 python slurm_test.py --epoch=200 --type=2

```

# Training time

|  # V100   | Training time (mins)  |
|  :----: | :----:  |
| 1  | 24.73 |
| 4  | 10.38 |
| 8  | 7.71 |
| 16  | 6.21 |

# Accuracy 

For multi-gpu (lr=0.01):
```
Epoch: [200]/[200], test accuracy: 77.040 %,  the best accuracy: 77.760 %
```

For single gpu (lr=0.01):
```
Epoch: [200]/[200], test accuracy: 79.030 %,  the best accuracy: 80.650 %
```

> Note:
> 1) If we fix args.seed, then all multi-gpu programs will produce the same results.
> 2) In comparison to a single gpu program, the batch size of a multi-gpu program increases, so the learning rate should also increase.
