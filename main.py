import argparse
import datetime
import os
import random
import subprocess

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
import torch.nn.functional as F
import pdb
import numpy as np
import torch.backends.cudnn as cudnn


starttime = datetime.datetime.now()

parser = argparse.ArgumentParser(description='')
# Training Configuration
parser.add_argument('--debug', type=int, default=0,
                    help='single gpu for code debug')
parser.add_argument('--seed', type=int, default=2022,
                    help='random seed')
parser.add_argument('--epoch', type=int, default=40)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=256,
                    help='total number of training iterations')
parser.add_argument('--data_dir', type=str, default='~/../share_data/zhangjie/')
parser.add_argument('--port', type=str, default='40', help='port, 0~65535')

args = parser.parse_args()


def setup_distributed(backend="nccl", port=None):
    """
    Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    import torch.distributed as dist
    
    """
    num_gpus = torch.cuda.device_count()

    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    node_list = os.environ["SLURM_NODELIST"]
    os.environ["MASTER_PORT"] = args.port
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank % num_gpus)
    os.environ["RANK"] = str(rank)
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )


def get_dataset():
    train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=False,
                                     transform=transforms.Compose(
                                         [
                                             transforms.RandomCrop(32, padding=4),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                         ]))

    test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=False,
                                    transform=transforms.Compose(
                                        [
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        ]))
    train_sampler, test_sampler = None, None
    if args.debug == 0:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
        shuffle=(train_sampler is None)
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, test_loader


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss

            total += data.shape[0]
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= total
    acc = 100. * correct / total
    return acc, test_loss

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True

if __name__ == "__main__":

    # 0. set up distributed device
    setup_seed(args.seed)
    rank = 0
    local_rank = args.gpu
    if args.debug == 0:
        setup_distributed()
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")
    else:
        print("Single gpu for debug >>> ")
    # 1. define network
    net = torchvision.models.resnet18(pretrained=False, num_classes=10)
    net = net.cuda(local_rank)
    # DistributedDataParallel
    if args.debug == 0:
        net = DDP(net, device_ids=[local_rank], output_device=local_rank)

    # 2. define dataloader
    train_loader, test_loader = get_dataset()
    # 3. define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=True, )

    if rank == 0:
        print(" =======  Training >>>>>> ======= \n")

    # 4. start to train
    net.train()
    best_acc = -1
    for ep in range(1, args.epoch + 1):
        train_loss = correct = total = 0
        # set sampler
        if args.debug == 0:
            train_loader.sampler.set_epoch(ep)

        for idx, (inputs, targets) in enumerate(train_loader):
            # pdb.set_trace()
            inputs, targets = inputs.to(local_rank), targets.to(local_rank)
            outputs = net(inputs)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total += targets.size(0)
            correct += torch.eq(outputs.argmax(dim=1), targets).sum().item()

        if rank == 0:
            print(
                "Epoch: [{}]/[{}], training accuracy: {:.3f} %, average training batch loss: {:.4f}".format(
                    ep + 1, args.epoch,
                    100.0 * correct / total, train_loss / total))
        acc, loss = test(net, test_loader)
        best_acc = max(best_acc, acc)
        if rank == 0:
            print("Epoch: [{}]/[{}], test accuracy: {:.3f} %,  the best accuracy: {:.3f} %".format(ep + 1,
                                                                                                   args.epoch, acc,
                                                                                                   best_acc))
    endtime = datetime.datetime.now()
    res = (endtime - starttime).seconds / 60  # mins
    if rank == 0:
        print("Total running time: {} mins!".format(res))

"""
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

"""
