import os
import argparse
from RCN_Trainer import RCN_Trainer

from tools.utils import *

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

# CEC 150 setp 40
def set_parameters(parser):
    # general setting
    parser.add_argument('--OSS', type=str2bool, default=False)
    parser.add_argument('--DLC', type=str2bool, default=False)
    parser.add_argument('--output_form_1', type=str, default='logit',
                        choices=['logit', 'softmax', 'scaled_softmax'],
                        help='The form of output given by model 1') #
    parser.add_argument('--output_form_2', type=str, default='logit',
                        choices=['logit', 'softmax', 'scaled_softmax'],
                        help='The form of output given by model 2') #
    parser.add_argument('--enable_auto_metric', type=str2bool, default=False)
    parser.add_argument('--intergrate_with_CEC', type=str2bool, default=False)
    parser.add_argument('--rotate_image', type=str2bool, default=True)
    parser.add_argument('--novel_sample_sysn_method', type=str, default='rotate') #
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--enable_meta_train', type=str2bool, default=True)
    parser.add_argument('--enable_global_loss', type=str2bool, default=True)
    parser.add_argument('--enable_local_loss', type=str2bool, default=True)
    parser.add_argument('--local_loss_weight', type=float, default=1.0)
    parser.add_argument('--global_loss_weight', type=float, default=1.0)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--storage_folder', type=str, default='tmpfile') # tmpfile
    parser.add_argument('--epoch', type=int, default=1,  # 50
                        help='base training epoches')
    parser.add_argument('--use_auto_scheduler', type=str2bool, default=False)
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--lr', type=float, default=0.1, # 0.1 
                        help='learning rate')
    parser.add_argument('--wd', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', type=str2bool, default=True)
    parser.add_argument('--scheduler', type=str, default='MSLR')
    parser.add_argument('--milestones', type=int, nargs='+', default=[30, 40, 60, 80]) # [30, 40]
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--steps', type=int, default=15) # 10

    # dataloader
    parser.add_argument('--state', type=str, default='train', choices=['train', 'test'],
                        help='training or testing')
    parser.add_argument('--network', type=str, default='ResNet18', choices=['ResNet18', 'ResNet20'],
                        help='Encoding and Decoding images')
    parser.add_argument('--dataset', type=str, default='cub_200',
                        choices=['miniImageNet', 'cifar_fs', 'cub_200', 'ImageNet_R'],
                        help='datasets')
    parser.add_argument('--used_data', type=str, default=None,
                        help='the name of csv file which used to train or test the model')
    parser.add_argument('--sampler', type=str, default=None,
                        help='data sampler')
    parser.add_argument('--train_val_sNode', type=int, default=None,#4
                        help='how many data in each category is used as the trainig data')
    parser.add_argument('--workers', type=int, default=8,#4
                        help='num of thread to process image')
    # pretrain setting
    parser.add_argument('--pretrain', type=str2bool, default=False)
    parser.add_argument('--joint', type=str2bool, default=False)
    parser.add_argument('--batch_size', type=int, default=128) #128
    parser.add_argument('--temperature', type=float, default=12) # 16

    # train_flag setting
    parser.add_argument('--train_flag', type=str2bool, default=False)
    parser.add_argument('--tasks', type=int, default=1)
    parser.add_argument('--n-way', type=int, default=10)
    parser.add_argument('--n-shot', type=int, default=5)
    parser.add_argument('--n-query', type=int, default=10)
    parser.add_argument('--val_method', type=str, default='fsl', choices=['single', 'db', 'fsl'])
    parser.add_argument('--sim_metric', type=str, default='cos', help="use cosine(cos) or euclidean(euc) to get sim")

    # test setting
    parser.add_argument('--batch_size_test', type=int, default=100)
    parser.add_argument('--test_method', type=str, default='pretrain')
    parser.add_argument('--test_model', type=str, default='default')

    # test setting
    parser.add_argument('--pseudo_epoch', type=int, default=100,  # 50
                        help='pseudo training epoches')
    parser.add_argument('--pseudo_batches', type=int, default=50,  # 50
                        help='tasks')
    parser.add_argument('--base_epoch', type=int, default=50,  # 50
                        help='base training epoches')

    # RCN setting
    parser.add_argument('--mode', type=str, default='cos')
    parser.add_argument('--trainer', type=str, default='default')
    parser.add_argument('--seed', type=int, default=5) # 
    parser.add_argument('--pretrained_model', type=str, default='std_pretrain') # 

    return parser

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='mynet')
    parser = set_parameters(parser)
    args = parser.parse_args()
    print(args)

    set_seed(args.seed)

    trainer = RCN_Trainer(args)
    # trainer.get_confusion_matrix()
    # trainer.get_data_for_TSNE()
    # trainer.topk_in_each_session()

    if args.pretrain:
        trainer.pretraining(joint=args.joint)
    else:
        if args.train_flag:
            trainer.train(reload=True)
        else:
            print('testing...')
            trainer.test(reload=True)
            # trainer.test_random_order(reload=True)
            # trainer.test_cross_domain(reload=True, cross_domain_set='cub_cifar')