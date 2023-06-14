from ast import arg
import os
import time
import torch
import numpy as np
from models.ResNet import resnet101, resnet18, resnet152, resnet34, resnet50
from models.resnet20_cifar import resnet20
from datasets import transform, sampler, datasets_loader
from collections import OrderedDict
import random
import torch.nn.functional as F
import re

from torch.nn.parallel import DistributedDataParallel as DDP

# from torch import distributed as dist
# def reduce_mean(tensor, nprocs):  # 用于平均所有gpu上的运行结果，比如loss
#     rt = tensor.clone()
#     dist.all_reduce(rt, op=dist.ReduceOp.SUM)
#     rt /= nprocs
#     return rt

def get_model_size(model):
    # 定义总参数量、可训练参数量及非可训练参数量变量
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    # 遍历model.parameters()返回的全局参数列表
    for param in model.parameters():
        mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
        Total_params += mulValue  # 总参数量
        if param.requires_grad:
            Trainable_params += mulValue  # 可训练参数量
        else:
            NonTrainable_params += mulValue  # 非可训练参数量

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')

def ddp(model, local_rank):
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    torch.distributed.barrier()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
    return DDP(model, device_ids=[local_rank], output_device=local_rank)


def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()

# calculate consumption time
class Timer:
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / float(p)
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(x / 60)
        return '{}s'.format(x)


# create folder to save result
def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# write contents
def log(log_file_path, string, mode='a+', whether_print=True):
    with open(log_file_path, mode) as f:
        f.write((string + '\n').encode('ascii', 'ignore').decode('ascii'))
        f.flush()
    if whether_print:
        print(string)


def one_hot(indices, depth):
    encode_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size() + torch.Size([1]))
    encode_indicies = encode_indicies.scatter_(1, index, 1)

    return encode_indicies


def count_accuracy(logits, label):
    pre = torch.argmax(logits, dim=1).view(-1)
    label = label.view(-1)
    accuracy = 100 * pre.eq(label).float().mean()
    return accuracy


def acc_topk(logits, label, k=1, return_ind_p=False):
    logits = F.softmax(logits, dim=-1)
    values, indices = torch.sort(logits, dim=-1, descending=True)
    if return_ind_p:
        return values[:, :k], indices[:, :k]
    sample_nums = logits.shape[0]
    topk_samples = 0
    for i in range(sample_nums):
        if label[i] in indices[i,:k]:
            topk_samples += 1
    return 100 * topk_samples / sample_nums


# get the feature extracting backbone and classification head
def get_model(args):
    if args.network == 'ResNet18':
        net = resnet18().cuda()
    elif args.network == 'ResNet20':
        net = resnet20().cuda()
    else:
        print("Invalid network type! in func:get_model")
        assert False

    return net

def get_dataset_imagenet(args, image_size: int=224):
    import torchvision.datasets as datasets
    transform_ = transform.image_augment(state=args.state, dataset=args.dataset, image_size=image_size)
    if args.state == 'train':
        path = args.imagenet_train_path
    elif args.state == 'test':
        path = args.imagenet_val_path
    else:
        AssertionError("Invalid state in get_dataset_imagenet")
    dataset = datasets.ImageFolder(path, transform_)
    return dataset


def get_dataset(args, dataset_min_cls: int=0, dataset_max_cls: int=10000, image_size: int=224):
    transform_ = transform.image_augment(state=args.state, dataset=args.dataset, image_size=image_size)
    cwd = os.getcwd()
    if args.dataset == 'miniImageNet':
        data_path = os.path.join(cwd, 'datasets/miniimagenet')
    elif args.dataset == 'cifar100':
        data_path = os.path.join(cwd, 'datasets/cifar100')
    elif args.dataset == 'cub_200':
        data_path = os.path.join(cwd, 'datasets/cub-200-2011')
    elif args.dataset == 'ImageNet':
        data_path = os.path.join(cwd, 'datasets/imagenet')
    elif args.dataset == 'ImageNet_R':
        data_path = os.path.join(cwd, 'datasets/imagenet-r')
    else:
        AssertionError(f"utils.get_dataset: incorrect name {args.dataset}")
    dataset = datasets_loader.CategoryDatasetFolder(data_path, 
                                                    args.used_data, 
                                                    transform_, 
                                                    dataset_min_cls=dataset_min_cls, 
                                                    dataset_max_cls=dataset_max_cls,
                                                    out_name=False)
    print("{} images in {} have been used for {}".format(dataset.__len__(), args.dataset, args.state))
    return dataset


def get_sampler(dataset, args, sample_info):
    sampler_type = args.sampler
    if sampler_type == 'std':# split data into n batches
        if args.world_size > 1:
            sampler_ = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            # sampler_ = torch.utils.data.BatchSampler(dataset, batch_size=args.batch_size, drop_last=False)
            sampler_ = None
    elif sampler_type == 'inc': # 
        sampler_ = sampler.inc_sampler(dataset.labels, *sample_info)
    elif sampler_type == 'inc_test': # 
        if args.dataset == 'cub_200' or args.dataset == 'ImageNet_R':
            sampler_ = sampler.Cub_Test_Sampler(dataset.labels, *sample_info)
        else:
            sampler_ = sampler.Batch_Sample_From_Range(dataset.labels, *sample_info)
    elif sampler_type == 'fsl':
        sampler_ = sampler.CategoriesSampler(dataset.labels, *sample_info)
    elif sampler_type == 'align_inc':
        sampler_ = sampler.CategoriesSamplerAlignInc(dataset.labels, *sample_info)
    elif sampler_type == 'autoTasks':
        sampler_ = sampler.Auto_Task_Sampler(dataset.labels, *sample_info)
    else: 
        AssertionError("Invalid sampler_type! sampler_type should be one of ['fsl', 'std', 'task', 'norep', 'inc', 'inc_test']")
    return sampler_


def get_dataloader(args, dataset_min_cls: int=0, dataset_max_cls: int=10000, image_size: int=224, sample_info=None):
    if args.dataset == 'ImageNet':
        dataset = get_dataset_imagenet(args)
    else:
        dataset = get_dataset(args, dataset_min_cls=dataset_min_cls, dataset_max_cls=dataset_max_cls, image_size=image_size)

    sampler_ = get_sampler(dataset, args, sample_info)

    if args.world_size > 1 and args.sampler == 'std':
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, sampler=sampler_)
    else:
        if args.sampler == 'std':
            loader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler_, num_workers=args.workers, pin_memory=True)
    return loader


def get_optimizer(model, args):
    # get optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, 
                                    momentum=args.momentum, weight_decay=args.wd, nesterov=args.nesterov)
    if args.scheduler is None:
        return optimizer
    # get scheduler
    if args.scheduler == 'SLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.steps, gamma=args.gamma)
    elif args.scheduler == 'MSLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    elif args.scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
    return optimizer, scheduler


# model structure relevant
def print_model_para_name(model):
    for name, param in model.named_parameters():
        print(name)


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def freeze_layer(model, layers_name_freeze=None, layer_name_keep=None):
    # backbone.conv1 backbone.bn1 backbone.layer1 .......
    if layers_name_freeze != None:
        for name, param in model.named_parameters():
            for fl in layers_name_freeze:
                if fl in name:
                    param.requires_grad = False
    if layer_name_keep != None:
        for name, param in model.named_parameters():
            for fl in layer_name_keep:
                if fl not in name:
                    param.requires_grad = False
    return model


def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def load_trained_paras(path: str, models: list, keys:list, map_location="cpu", logger=None, sub_level=None):
    if logger:
        logger.info(f"Load pretrained model [{model.__class__.__name__}] from {path}")
    if os.path.exists(path):
        # From local
        state_dict = torch.load(path, map_location)
    # elif path.startswith("http"):
    #     # From url
    #     state_dict = load_state_dict_from_url(path, map_location=map_location, check_hash=False)
    else:
        raise Exception(f"Cannot find {path} when load pretrained")
    
    model_trained = []
    for i in range(len(models)):
        model, key = models[i], keys[i]
        model = load_pretrained_dict(model, state_dict, key)
        model_trained.append(model)

    return model_trained


def _auto_drop_invalid(model: torch.nn.Module, state_dict: dict, logger=None):
    """ Strip unmatched parameters in state_dict, e.g. shape not matched, type not matched.

    Args:
        model (torch.nn.Module):
        state_dict (dict):
        logger (logging.Logger, None):

    Returns:
        A new state dict.
    """
    ret_dict = state_dict.copy()
    invalid_msgs = []
    for key, value in model.state_dict().items():
        if key in state_dict:
            # Check shape
            new_value = state_dict[key]
            if value.shape != new_value.shape:
                invalid_msgs.append(f"{key}: invalid shape, dst {value.shape} vs. src {new_value.shape}")
                ret_dict.pop(key)
            elif value.dtype != new_value.dtype:
                invalid_msgs.append(f"{key}: invalid dtype, dst {value.dtype} vs. src {new_value.dtype}")
                ret_dict.pop(key)
    if len(invalid_msgs) > 0:
        warning_msg = "ignore keys from source: \n" + "\n".join(invalid_msgs)
        if logger:
            logger.warning(warning_msg)
        else:
            import warnings
            warnings.warn(warning_msg)
    return ret_dict


def load_pretrained_dict(model: torch.nn.Module, state_dict: dict, key:str, logger=None, sub_level=None):
    """ Load parameters to model with
    1. Sub name by revise_keys For DataParallelModel or DistributeParallelModel.
    2. Load 'state_dict' again if possible by key 'state_dict' or 'model_state'.
    3. Take sub level keys from source, e.g. load 'backbone' part from a classifier into a backbone model.
    4. Auto remove invalid parameters from source.
    5. Log or warning if unexpected key exists or key misses.

    Args:
        model (torch.nn.Module):
        state_dict (dict): dict of parameters
        logger (logging.Logger, None):
        sub_level (str, optional): If not None, parameters with key startswith sub_level will remove the prefix
            to fit actual model keys. This action happens if user want to load sub module parameters
            into a sub module model.
    """
    revise_keys = [(r'^module\.', '')]
    state_dict = state_dict[key]
    for p, r in revise_keys:
        state_dict = {re.sub(p, r, k): v for k, v in state_dict.items()}

    if sub_level:
        sub_level = sub_level if sub_level.endswith(".") else (sub_level + ".")
        sub_level_len = len(sub_level)
        state_dict = {key[sub_level_len:]: value
                      for key, value in state_dict.items()
                      if key.startswith(sub_level)}

    state_dict = _auto_drop_invalid(model, state_dict, logger=logger)

    load_status = model.load_state_dict(state_dict, strict=False)
    unexpected_keys = load_status.unexpected_keys
    missing_keys = load_status.missing_keys
    err_msgs = []
    if unexpected_keys:
        err_msgs.append('unexpected key in source '
                        f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msgs.append('missing key in source '
                        f'state_dict: {", ".join(missing_keys)}\n')
    err_msgs = '\n'.join(err_msgs)

    if len(err_msgs) > 0:
        if logger:
            logger.warning(err_msgs)
        else:
            import warnings
            warnings.warn(err_msgs)
    return model