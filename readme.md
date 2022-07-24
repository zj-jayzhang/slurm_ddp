
# Introduction

This is an example of using the Slurm cluster to speed up the Torch model training.


# Slurm Cluster Usage

usage: set args.debug = 0 for slurm, 1 for single gpu

Here is an example on V100:
``` python
# slurm, 1 gpu for debug, slurm will randomly seletc one node, no need to set the gpu

srun --mpi=pmi2 -p v100_nodes -n 1 --gres=gpu:1 --ntasks-per-node=1  --job-name=test --kill-on-bad-exit=1 python main.py --port=29500 --epoch=200 --debug=1 


# slurm, 16 gpu on two nodes (8 V100 for each node)

srun --mpi=pmi2 -p v100_nodes -n 16 --gres=gpu:8 --ntasks-per-node=8  --job-name=test --kill-on-bad-exit=1 python main.py --port=29499 --epoch=200

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
