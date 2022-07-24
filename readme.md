usage: debug = 0 for slurm, 1 for single gpu



1) slurm, 1 gpu for debug, slurm will randomly seletc one node, no need to set the gpu
srun --mpi=pmi2 -p stc1_v100_16g -n 1 --gres=gpu:1 --ntasks-per-node=1  --job-name=test --kill-on-bad-exit=1 -w SH-IDC1-10-5-37-69 python slurm_train.py --p
ort=29500 --epoch=200 --debug=1 | tee single.log 

Epoch: [200]/[200], test accuracy: 79.030 %,  the best accuracy: 80.650 %
Total running time: 24.733333333333334 mins!

2) slurm, 4 gpu 
srun --mpi=pmi2 -p stc1_v100_16g -n 4 --gres=gpu:4 --ntasks-per-node=4  --job-name=test --kill-on-bad-exit=1 -x SH-IDC1-10-5-36-96 python -u slurm_train.py --port=29499 --epoch=200

Epoch: [200]/[200], test accuracy: 78.040 %,  the best accuracy: 78.040 %
Total running time: 10.383333333333333 mins!

3) slurm, 8 gpu
srun --mpi=pmi2 -p stc1_v100_16g -n 8 --gres=gpu:8 --ntasks-per-node=8  --job-name=test --kill-on-bad-exit=1 -x SH-IDC1-10-5-36-96 python -u slurm_train.py --port=29498 --epoch=200

Epoch: [200]/[200], test accuracy: 76.480 %,  the best accuracy: 77.120 %
Total running time: 7.716666666666667 mins!

4) slurm, 16 gpu on two nodes 
srun --mpi=pmi2 -p stc1_v100_16g -n 16 --gres=gpu:8 --ntasks-per-node=8  --job-name=test --kill-on-bad-exit=1 -x SH-IDC1-10-5-36-96 python -u slurm_train.py --port=29499 --epoch=200

Epoch: [200]/[200], test accuracy: 74.400 %,  the best accuracy: 78.400 %
Total running time: 6.2 mins!