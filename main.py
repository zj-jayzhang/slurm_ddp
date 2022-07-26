import argparse
import datetime
import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
import pdb
import numpy as np
import torch.backends.cudnn as cudnn
import warnings

warnings.filterwarnings('ignore')

starttime = datetime.datetime.now()

parser = argparse.ArgumentParser(description='')
# Training Configuration
parser.add_argument('--type', type=int, default=0,
                    help='0 for single gpu, 1 for torch.distributed.launch, 2 for slurm.')
parser.add_argument('--seed', type=int, default=2022,
                    help='random seed')
parser.add_argument('--epoch', type=int, default=40)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=256,
                    help='total number of training iterations')
parser.add_argument('--data_dir', type=str, default='~/../share_data/zhangjie/')
parser.add_argument('--port', type=str, default='40', help='port, 0~65535')
parser.add_argument('--eval', type=int, default=0, help='test the performance of the pretrained model')


args = parser.parse_args()

def setup_distributed(backend="nccl", port=None):
    """
    Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    import torch.distributed as dist
    
    """
    num_gpus = torch.cuda.device_count()
    if args.type == 2:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        os.environ["MASTER_PORT"] = '36666'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    elif args.type == 1:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

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
    if args.type != 0:
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
    if args.type == 0:
        print("Single gpu running >>> ")
    else:
        setup_distributed()
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])

        print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")

    # 1. define network
    net = torchvision.models.resnet18(pretrained=False, num_classes=10)
    net = net.cuda(local_rank)
    # DistributedDataParallel
    if args.type != 0:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank)

    # 2. define dataloader
    train_loader, test_loader = get_dataset()
    if args.eval == 1:
        checkpoint = torch.load('ckpt.pth', map_location="cpu")
        net.module.load_state_dict(checkpoint)
        test_acc, _ = test(net, test_loader)
        if rank == 0:   print("Test accuracy: {}".format(test_acc))
        os._exit(0) 

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
        if args.type != 0:
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
        if acc > best_acc:
            best_acc = acc
            if rank == 0: torch.save(net.module.state_dict(), 'ckpt.pth')
        if rank == 0:
            print("Epoch: [{}]/[{}], test accuracy: {:.3f} %,  the best accuracy: {:.3f} %".format(ep + 1,
                                                                                                   args.epoch, acc,
                                                                                                   best_acc))
    endtime = datetime.datetime.now()
    res = (endtime - starttime).seconds / 60  # mins
    if rank == 0:
        print("Total running time: {} mins!".format(res))

"""
usage: 0 for single gpu, 1 for torch.distributed.launch, 2 for slurm.

# torch.distributed.launch
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=22222 ddp.py --epoch=100 --type=1 --data_dir=data/

# eval
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=22222 ddp.py --type=1 --data_dir=data/ --eval=1

# slurm, 4 gpu on 1 node

srun --mpi=pmi2 -p stc1_v100_32g -n 4 --gres=gpu:4 --ntasks-per-node=4  --job-name=test --kill-on-bad-exit=1 python slurm_test.py --epoch=200 --type=2

# slurm, 16 gpu on two nodes (8 V100 for each node)

srun --mpi=pmi2 -p stc1_v100_32g -n 16 --gres=gpu:8 --ntasks-per-node=8  --job-name=test --kill-on-bad-exit=1 python slurm_test.py --epoch=200 --type=2


"""
