import os
from tqdm import tqdm
import torch.nn.functional as F

import torch.nn as nn
from copy import deepcopy

from tools.utils import *
from models.MyModel import MyModel
from models import ProtoNet

import math

import pdb

import numpy as np

class RCN_Trainer():
    def __init__(self, args):
        self.args = args
        # initialize model
        self.model      = MyModel(self.args).cuda()
        self.fsl_model  = ProtoNet(self.model.num_features, 
                                   self.model.num_cls, 
                                   enable_auto_metric=self.args.enable_auto_metric).cuda()

        # initialize save dir
        self.cwd = os.getcwd()
        if self.args.DLC:
            self.cwd = self.cwd.replace('/root/code/fscil', '/root/data/wy/FSIL') 
        
        self.init_paths()

        print("Current work dir is {}".format(self.work_dir))
        check_dir(self.work_dir)
        self.model_save_name = 'PN.pth' #
        self.model_save_key  = 'PN' # 

        self.inc_shot        = 5

        # performance records
        self.performance_stati = {
            'details': torch.zeros(self.model.sessions, self.model.sessions), # acc on each session
            'forgetting': 0,
            'acc_base': 0,
            'acc_novel': 0,
            'acc_each_sess': torch.zeros(self.model.sessions)
        }


    def get_train_loader(self, 
                         sampler_type: str='std', 
                         joint: bool=False, 
                         transform_state: str='train',
                         full_data: bool=False,
                         cur_session: int=0):
        """
        @ sampler_type: sample batch if 'std' or sample task  if 'fsl'
        """
        if joint:
            self.args.used_data = self.data_prefix+'joint'
        else:
            self.args.used_data = self.data_prefix+'base'

        if full_data:
            self.args.used_data = self.data_prefix+'full'

        self.args.state     = transform_state
        self.args.sampler   = sampler_type

        min_cls = 0 if (cur_session == 0 or joint) else self.model.base_cls_num + (cur_session-1) * self.model.inc_cls_num
        max_cls = self.model.base_cls_num + cur_session * self.model.inc_cls_num
        if sampler_type == 'fsl':
            sample_info=[self.args.tasks, 1, self.args.n_way, self.args.n_shot, self.args.n_query, max_cls, min_cls]
            train_loader = get_dataloader(self.args, dataset_min_cls=min_cls, dataset_max_cls=max_cls, sample_info=sample_info)
        elif sampler_type == 'align_inc':
            sample_info=[self.args.tasks, 1, self.args.n_way, self.args.n_shot, self.args.n_query, max_cls, min_cls, self.args.batch_size]
            train_loader = get_dataloader(self.args, dataset_min_cls=min_cls, dataset_max_cls=max_cls, sample_info=sample_info)
        else:
            train_loader = get_dataloader(self.args, dataset_min_cls=min_cls, dataset_max_cls=max_cls)

        return train_loader

    # for train-free method, sample all incremental sessions
    def get_inc_loader(self, 
                       ways: int=10,  
                       shots: int=5, 
                       max_cls: int=10000, 
                       min_cls:int=0):
        self.args.state = 'test'
        self.args.used_data = self.data_prefix + 'inc'
        if self.inc_shot > 5:
            self.args.used_data = self.data_prefix + 'full'
        self.args.sampler = 'inc'

        inc_train_loader = get_dataloader(self.args, 
                                          dataset_min_cls=min_cls, 
                                          dataset_max_cls=max_cls,
                                          sample_info=[ways, shots, max_cls, min_cls])

        return inc_train_loader


    def get_test_loader(self, num_seen_cls, min_cls:int=0):
        self.args.state = 'test'
        self.args.used_data = self.data_prefix+'test'
        self.args.sampler = 'inc_test'
        test_loader = get_dataloader(self.args, sample_info=[self.args.batch_size_test, num_seen_cls, min_cls])
        
        return test_loader


    def get_unique_label(self, label):
        uni_label = []
        for l in label:
            if l not in uni_label:
                uni_label.append(l)
        return uni_label


    def get_optim(self):
        optimizer = torch.optim.SGD([{'params': self.fsl_model.parameters()}
                                    ], lr=self.args.lr, momentum=self.args.momentum,
                                    weight_decay=self.args.wd, nesterov=self.args.nesterov)

        if self.args.scheduler == 'SLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                                    optimizer, step_size=self.args.steps, gamma=self.args.gamma)
        elif self.args.scheduler == 'MSLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                                    optimizer, milestones=self.args.milestones, gamma=self.args.gamma)
        return optimizer, scheduler


    def train(self, reload=True):
        """
        @ para use_auto_scheduler: whether use auto strategy to update lr
        @ para test_method: the test method, eg: fsl_test or test choice=['fsl', 'single', 'db]
        @ para global_loss: align the test process
        @ para reload: whether reload previous model
        """
        n_w, n_s, n_q   = self.args.n_way, self.args.n_shot, self.args.n_query
        num_sup = n_w * n_s
        # initialize log
        fs_train_log    = os.path.join(self.work_dir, 'train.log')
        model_save_path = os.path.join(self.work_dir, self.model_save_name)
        log(fs_train_log, str(vars(self.args)))

        # initialize backbone
        if reload: 
            self.model = load_trained_paras(self.pre_model_path, [self.model], ['pretrain'])[0]
        self.fsl_model.backbone = deepcopy(self.model.backbone) 

        # few-shot train loader
        if self.args.enable_meta_train:
            train_loader = self.get_train_loader('fsl')
        else:
            train_loader = self.get_train_loader()

        # initilize optimizer
        optimizer, scheduler = self.get_optim()
    
        # training
        timer = Timer()
        max_val_acc = 0.0
        eps = 0.1
        sup_label = torch.arange(n_w).repeat(n_s).reshape(-1).type(torch.cuda.LongTensor)
        que_label = torch.arange(n_w).repeat(n_q).reshape(-1).type(torch.cuda.LongTensor)

        if self.args.novel_sample_sysn_method != 'none':
            total_w = 2*n_w
            sup_label_cat   = torch.cat((sup_label, sup_label + n_w), dim=0)
            que_label_cat   = torch.cat((que_label, que_label + n_w), dim=0)
        else:
            total_w = n_w
            sup_label_cat   = deepcopy(sup_label)
            que_label_cat   = deepcopy(que_label)
        smoothed_one_hot    = one_hot(que_label_cat, total_w)
        smoothed_one_hot    = smoothed_one_hot * (1 - eps) + (1 - smoothed_one_hot) * eps / (total_w - 1)

        for epoch in range(1, self.args.epoch+1):
            log(fs_train_log, 'Session:{}\tTrain Epoch:{}\tLearning Rate:{:.6f}'.format(0, epoch, scheduler.get_last_lr()[0]))
            
            self.fsl_model.train()

            train_loss = []
            for i, batch in enumerate(tqdm(train_loader)):
                # comput loss and acc
                data, true_label = [_.cuda() for _ in batch]
                loss = 0.0
                if not self.args.enable_meta_train:
                    feats   = self.fsl_model.encode(data)
                    proto   = self.fsl_model.fc.weight[:self.model.base_cls_num]
                    logits  = self.fsl_model.compute_similarity(proto, feats, mode=self.args.sim_metric)
                    if self.args.enable_global_loss:
                        feat_bs = self.model.encode(data)
                        logits += self.fsl_model.compute_similarity(
                                self.model.fc.weight[:self.model.base_cls_num], feat_bs, mode='cos')
                    loss    = F.cross_entropy(self.args.temperature * logits, true_label)
                    acc     = count_accuracy(logits, true_label)
                else:
                    sup_data, que_data = data[:num_sup], data[num_sup:]
                    sup_data_ori, que_data_ori = deepcopy(sup_data), deepcopy(que_data)
                    sup_gt, que_gt  = true_label[:num_sup], true_label[num_sup:]

                    sup_imgs_tmp = None
                    if self.args.novel_sample_sysn_method == 'rotate':
                        sup_imgs_tmp, que_imgs_tmp = self.rotate_image(deepcopy(sup_data), deepcopy(que_data))

                    elif self.args.novel_sample_sysn_method == 'mixup_inter':
                        sup_imgs_tmp, que_imgs_tmp = self.augment(deepcopy(sup_data), 
                                                                  deepcopy(que_data),
                                                                  alpha=self.args.alpha,
                                                                  mtype='mixup_inter')

                    elif self.args.novel_sample_sysn_method == 'mixup_intra':
                        sup_imgs_tmp, que_imgs_tmp = self.augment(deepcopy(sup_data), 
                                                                deepcopy(que_data),
                                                                alpha=self.args.alpha,
                                                                mtype='mixup_intra')

                    elif self.args.novel_sample_sysn_method == 'cutmix_inter':
                        sup_imgs_tmp, que_imgs_tmp = self.augment(deepcopy(sup_data), 
                                                                deepcopy(que_data),
                                                                alpha=self.args.alpha,
                                                                mtype='cutmix_inter')

                    elif self.args.novel_sample_sysn_method == 'cutmix_intra':
                        sup_imgs_tmp, que_imgs_tmp = self.augment(deepcopy(sup_data), 
                                                                deepcopy(que_data),
                                                                alpha=self.args.alpha,
                                                                mtype='cutmix_intra')
                   
                    if sup_imgs_tmp is not None:
                        sup_data = torch.cat((sup_data, sup_imgs_tmp), dim=0)
                        que_data = torch.cat((que_data, que_imgs_tmp), dim=0)

                    if self.args.enable_local_loss:
                        logits, fs_feature = self.fsl_model(sup_data, sup_label_cat, que_data, n_way=total_w, mode=self.args.sim_metric, return_feat=True)
                        log_prob    = F.log_softmax(self.args.temperature * logits.reshape(-1, total_w), dim=1)
                        loss        = -self.args.local_loss_weight * (log_prob * smoothed_one_hot).sum(dim=1).mean()

                    if self.args.enable_global_loss:
                        if self.args.novel_sample_sysn_method != 'none':
                            sup_fs, que_fs  = fs_feature[:num_sup], fs_feature[num_sup:]
                        else:
                            sup_fs = self.fsl_model.encode(sup_data)
                            que_fs = self.fsl_model.encode(que_data)  
                        proto_          = sup_fs.reshape(n_s, n_w, -1).mean(dim=0)
                        proto_fs        = self.fsl_model.fc.weight[:self.model.base_cls_num].clone().detach()
                        idx             = self.get_unique_label(sup_gt)
                        idx             = [ind.item() for ind in idx]
                        proto_fs[idx]   = proto_
                        logit_fs        = self.fsl_model.compute_similarity(proto_fs, que_fs, mode=self.args.sim_metric)
                        if self.args.enable_auto_metric:
                            logit_fs = F.softmax(logit_fs, dim=-1)

                        que_bs        = self.model.encode(que_data_ori)
                        sup_bs        = self.model.encode(sup_data_ori)
                        proto_bs      = self.model.fc.weight[:self.model.base_cls_num].clone().detach()
                        proto_bs[idx] = sup_bs.reshape(n_s, n_w, -1).mean(dim=0)

                        logit_bs      = self.fsl_model.compute_similarity(proto_bs, que_bs, mode='cos') # 60.47
                        # logit_bs      = F.linear(F.normalize(que_bs, p=2, dim=-1), F.normalize(proto_bs, p=2, dim=-1))

                        if self.args.enable_auto_metric:
                            logit_bs  = F.softmax(logit_bs, dim=-1)
                        
                        if 'softmax' in self.args.output_form_1:
                            logit_bs = F.softmax(logit_bs, dim=-1)

                        if 'softmax' in self.args.output_form_2:
                            logit_fs = F.softmax(logit_fs, dim=-1)

                        logits_       = logit_fs + logit_bs
                        loss          += self.args.global_loss_weight * F.cross_entropy(self.args.temperature*logits_, que_gt)

                train_loss.append(loss.item())

                # update paras
                optimizer.zero_grad();loss.backward();optimizer.step()

                # output log
                if i % 10 == 0:
                    train_loss_avg = np.mean(np.array(train_loss))
                    log(fs_train_log,
                        'Train Epoch:{}\tBatch:[{}/{}]\tLoss:{:.4f} % ({:.4f})'.format(
                            epoch, i, len(train_loader), train_loss_avg, loss.item()
                        ))

            scheduler.step()

            # validation
            val_acc_avg = self.test()

            log(fs_train_log, 'Validation Epoch:{}\tAccuracy:{:.2f}'.format(epoch, val_acc_avg))
            # save model
            if val_acc_avg > max_val_acc:
                max_val_acc = val_acc_avg
                torch.save({self.model_save_key: self.fsl_model.state_dict()}, model_save_path)  

            log(fs_train_log, 'Elapsed Time: {}/{}\n'.format(
                                            timer.measure(), timer.measure(epoch / float(81))))
        return 0


    def single_session_test(self, cur_session):

        num_seen_cls  = self.model.base_cls_num + cur_session * self.model.inc_cls_num

        # get test_loader
        inc_test_loader = self.get_test_loader(num_seen_cls)

        test_accs, collect_preds, collect_labels = [], [], []

        feature_base, feature_fs, label_all = [], [], []
        # testing
        with torch.no_grad():
            for i, batch in enumerate(tqdm(inc_test_loader)):
                data, label = [_.cuda() for _ in batch]
                x_base      = self.model.encode(data)
                x_fsl       = self.fsl_model.encode(data)

                if cur_session == self.model.sessions-1:
                    feature_base.append(x_base)
                    feature_fs.append(x_fsl)
                    label_all.append(label)

                if self.args.test_method == 'pretrain':
                    proto_base  = self.model.fc.weight[:num_seen_cls]
                    logits_base = F.linear(F.normalize(x_base, p=2, dim=-1), F.normalize(proto_base, p=2, dim=-1)) 
                    scores_     = torch.softmax(logits_base, dim=-1)
                    preds       = torch.argmax(scores_, dim=-1).reshape(-1)

                else:
                    # score given by base
                    proto_base = self.model.fc.weight[:num_seen_cls]
                    logit_base = F.linear(F.normalize(x_base, p=2, dim=-1), F.normalize(proto_base, p=2, dim=-1))
                    # logit_base = self.fsl_model.compute_similarity(proto_base, x_base, mode='cos') wrong because d
        
                    if self.args.output_form_1 == 'softmax':
                        score_base = torch.softmax(logit_base, dim=-1)
                    elif self.args.output_form_1 == 'scaled_softmax':
                        score_base = torch.softmax(self.args.temperature * logit_base, dim=-1)
                    else:
                        score_base = logit_base

                    # score_base = 0

                    # score given by fsl
                    proto_fsl = self.fsl_model.fc.weight[:num_seen_cls]
                    if 'cos' in self.args.sim_metric:
                        logit_fsl = F.linear(F.normalize(x_fsl, p=2, dim=-1), F.normalize(proto_fsl, p=2, dim=-1))
                    else:
                        logit_fsl = self.fsl_model.compute_similarity(proto_fsl, x_fsl, mode=self.args.sim_metric)

                    if self.args.output_form_2 == 'softmax':
                        score_fsl = torch.softmax(logit_fsl, dim=-1)
                    elif self.args.output_form_2 == 'scaled_softmax':
                        score_fsl = torch.softmax(self.args.temperature * logit_fsl, dim=-1)
                    else:
                        score_fsl = logit_fsl

                    # score_fsl = 0

                    # model ensemble
                    score_ens = (score_base + score_fsl) / 2
                    score, preds = torch.max(score_ens, dim=-1)

                accuracy = 100 * preds.eq(label).float().mean()
                test_accs.append(accuracy.item())
                collect_preds.append(preds)
                collect_labels.append(label)

        collect_preds   = torch.cat(collect_preds, dim=0)
        collect_labels  = torch.cat(collect_labels, dim=0)
        # if cur_session == self.model.sessions-1:
        #     # for confusion matrix
        #     pds = collect_preds.cpu().numpy()
        #     gts = collect_labels.cpu().numpy()
        #     np.savez('PlotFactory/RCN/CIFAR_Ours.npz', pds=pds, gts=gts)

        if cur_session == self.model.sessions-1:
            feature_base = torch.cat(feature_base, 0)
            feature_fs = torch.cat(feature_fs, 0)
            label_all = torch.cat(label_all, 0).reshape(-1)
            
            protos_base = self.model.fc.weight.detach().cpu().numpy()
            protos_fs = self.fsl_model.fc.weight.detach().cpu().numpy()
            feature_base = feature_base.detach().cpu().numpy()
            feature_fs = feature_fs.detach().cpu().numpy()
            label_all = label_all.detach().cpu().numpy()

            np.savez('PlotFactory/RCN/data/tsne_base.npz', 
                    proto=protos_base, feature=feature_base, label=label_all)

            np.savez('PlotFactory/RCN/data/tsne_fsl.npz',
                    proto=protos_fs, feature=feature_fs, label=label_all)

        # performance on each session
        self.performance_analysis(cur_session, collect_preds, collect_labels)
        return np.mean(np.array(test_accs)), 1.96 * np.std(np.array(test_accs)) / np.sqrt(len(test_accs))


    def test(self, reload=False):
        test_log = os.path.join(self.work_dir, 'test_' + self.args.test_method +'.log')

        if reload:
            self.model = load_trained_paras(self.pre_model_path, [self.model], ['pretrain'])[0]
            # MN
            # mn_path = 'experiments/RCN/mini/mini_15w20s_MN/PN.pth'
            # self.fsl_model = load_trained_paras(mn_path, [self.fsl_model], [self.model_save_key])[0]
            # self.model.backbone = deepcopy(self.fsl_model.backbone)
            # self.model  = self.update_base_fc(self.model)

            # self.model  = self.update_base_fc(self.model)
            if self.args.test_method != 'pretrain':
                fs_model_path   = os.path.join(self.work_dir, self.model_save_name)
                self.fsl_model  = load_trained_paras(fs_model_path, [self.fsl_model], [self.model_save_key])[0]
        self.fsl_model  = self.update_base_fc(self.fsl_model)

        # incremental train loader
        inc_train_loader = self.get_inc_loader(self.model.inc_cls_num, 
                                                self.inc_shot, 
                                                max_cls=self.model.num_cls,
                                                min_cls=self.model.base_cls_num)

        self.model.eval();self.fsl_model.eval()
        # session 0
        acc, _ = self.single_session_test(0)
        self.performance_stati['acc_each_sess'][0] = acc
        log(test_log, 'Sess:{}\tAcc:{:.2f}'.format(0, acc))
        # session > 1
        with torch.no_grad():
            for sess, batch in enumerate(tqdm(inc_train_loader)):
                data, label     = [_.cuda() for _ in batch]
                start_label     = self.model.base_cls_num + sess * self.model.inc_cls_num
                end_label       = self.model.base_cls_num + (sess+1) * self.model.inc_cls_num
                cls_list        = np.arange(start_label, end_label)
                self.model      = self.update_inc_fc(self.model, data, label, cls_list)
                self.fsl_model  = self.update_inc_fc(self.fsl_model, data, label, cls_list)

                acc, acc_aci95= self.single_session_test(sess+1)
                self.performance_stati['acc_each_sess'][sess+1] = acc
                log(test_log, 'Sess:{}\tAcc:{:.2f}'.format(sess+1, acc))

        log(test_log, 'Performance info:{}'.format(self.performance_stati))
        log(test_log, 'Average: {}'.format(self.performance_stati['acc_each_sess'].mean()))
        return acc


    def single_session_test_random_order(self, cur_session: int=0, mode='inc', return_mode='only_acc', memory=None): 

        # testing
        test_accs, collect_preds, collect_labels = [], [], []

        feature_base, feature_fs, label_all = [], [], []
        num_seen_cls = self.model.base_cls_num+cur_session*self.model.inc_cls_num

        # get test data
        indx    = deepcopy(self.order[:cur_session])
        datas   = deepcopy(self.datas_inc_test[str(0)])
        labels  = deepcopy(self.labels_inc_test[str(0)])

        sp = 60
        for j, ind in enumerate(indx):
            ind = ind.item()+1
            new_data  = self.datas_inc_test[str(ind)]
            new_label = self.labels_inc_test[str(ind)]

            gt_label_uni  = range(self.model.base_cls_num+(ind-1)*self.model.inc_cls_num
                                ,self.model.base_cls_num+ind*self.model.inc_cls_num)
            rel_label_uni = range(sp, sp + self.model.inc_cls_num)
            label_map     = dict(zip(gt_label_uni, rel_label_uni))

            for i, data in enumerate(new_data):
                datas.append(data)
                label           = new_label[i]
                label           = [x.item() for x in label]
                mapped_labels   = [label_map[x] for x in label]
                label           = torch.LongTensor(mapped_labels).cuda()
                labels.append(label)
            sp = sp + self.model.inc_cls_num
        
        with torch.no_grad():
            for i, data in enumerate(tqdm(datas)):
                label = labels[i]
                x_base      = self.model.encode(data)
                x_fsl       = self.fsl_model.encode(data)

                if cur_session == 8:
                    feature_base.append(x_base)
                    feature_fs.append(x_fsl)
                    label_all.append(label)
                if self.args.test_method == 'pretrain':
                    proto_base  = self.model.fc.weight[:num_seen_cls]
                    logits_base = F.linear(F.normalize(x_base, p=2, dim=-1), F.normalize(proto_base, p=2, dim=-1)) 
                    scores_     = torch.softmax(logits_base, dim=-1)
                    preds       = torch.argmax(scores_, dim=-1).reshape(-1)

                else:
                    # score given by base
                    proto_base = self.model.fc.weight[:num_seen_cls]
                    logit_base = F.linear(F.normalize(x_base, p=2, dim=-1), F.normalize(proto_base, p=2, dim=-1))
                    # logit_base = self.fsl_model.compute_similarity(proto_base, x_base, mode='cos') wrong because d
        
                    if self.args.output_form_1 == 'softmax':
                        score_base = torch.softmax(logit_base, dim=-1)
                    elif self.args.output_form_1 == 'scaled_softmax':
                        score_base = torch.softmax(self.args.temperature * logit_base, dim=-1)
                    else:
                        score_base = logit_base

                    # score_base = 0

                    # score given by fsl
                    proto_fsl = self.fsl_model.fc.weight[:num_seen_cls]
                    if 'cos' in self.args.sim_metric:
                        logit_fsl = F.linear(F.normalize(x_fsl, p=2, dim=-1), F.normalize(proto_fsl, p=2, dim=-1))
                    else:
                        logit_fsl = self.fsl_model.compute_similarity(proto_fsl, x_fsl, mode=self.args.sim_metric)

                    if self.args.output_form_2 == 'softmax':
                        score_fsl = torch.softmax(logit_fsl, dim=-1)
                    elif self.args.output_form_2 == 'scaled_softmax':
                        score_fsl = torch.softmax(self.args.temperature * logit_fsl, dim=-1)
                    else:
                        score_fsl = logit_fsl

                    # score_fsl = 0

                    # model ensemble
                    score_ens = (score_base + score_fsl) / 2
                    score, preds = torch.max(score_ens, dim=-1)

                accuracy = 100 * preds.eq(label).float().mean()
                test_accs.append(accuracy.item())
                collect_preds.append(preds)
                collect_labels.append(label)

            # performance matrix
            collect_preds = torch.cat(collect_preds, dim=0)
            collect_labels = torch.cat(collect_labels, dim=0)
            # if cur_session == 10:
            #     pds = collect_preds.cpu().numpy()
            #     gts = collect_labels.cpu().numpy()
            #     np.savez('PlotFactory/REPO/CEC.npz', pds=pds, gts=gts)
            if self.model.inc_cls_num != 1:
                self.performance_analysis(cur_session, collect_preds, collect_labels)
            
            # for confusion matrix
            return np.mean(np.array(test_accs))


    def test_random_order(self, reload=False):

        test_log = os.path.join(self.work_dir, 'test_' + self.args.test_method +'.log')

        if reload:
            self.model = load_trained_paras(self.pre_model_path, [self.model], ['pretrain'])[0]
            if self.args.test_method != 'pretrain':
                fs_model_path   = os.path.join(self.work_dir, self.model_save_name)
                self.fsl_model  = load_trained_paras(fs_model_path, [self.fsl_model], [self.model_save_key])[0]
        self.fsl_model  = self.update_base_fc(self.fsl_model)
        self.model.eval();self.fsl_model.eval()

        # incremental train loader
        inc_train_loader = self.get_inc_loader(self.model.inc_cls_num, 
                                                self.inc_shot, 
                                                max_cls=self.model.num_cls,
                                                min_cls=self.model.base_cls_num)

        self.model.eval();self.fsl_model.eval()
        # session 0
        acc, _ = self.single_session_test(0)
        self.performance_stati['acc_each_sess'][0] = acc
        log(test_log, 'Sess:{}\tAcc:{:.2f}'.format(0, acc))

        ##########################
        # store data
        datas  = []
        labels = []

        # stage test data
        self.datas_inc_test = {}
        self.labels_inc_test = {}

        # split 0
        test_loader = self.get_test_loader(60)
        splits      = str(0)
        if splits not in self.datas_inc_test:
            self.datas_inc_test[splits] = []
            self.labels_inc_test[splits] = []
        for i, batch in enumerate(tqdm(test_loader)):
            data, label = [_.cuda() for _ in batch]
            self.datas_inc_test[splits].append(data)
            self.labels_inc_test[splits].append(label)

        for sess, batch in enumerate(tqdm(inc_train_loader)):
            # stage inc train loader
            data, label = [_.cuda() for _ in batch]
            datas.append(data)
            labels.append(label)

            # split i
            num_seen_cls = self.model.base_cls_num + (sess+1) * self.model.inc_cls_num
            test_loader = self.get_test_loader(num_seen_cls, num_seen_cls-5)
            splits = str(sess+1)
            if splits not in self.datas_inc_test:
                self.datas_inc_test[splits] = []
                self.labels_inc_test[splits] = []

            for i, batch in enumerate(tqdm(test_loader)):
                data, label = [_.cuda() for _ in batch]
                self.datas_inc_test[splits].append(data)
                self.labels_inc_test[splits].append(label)

        torch.manual_seed(1314111)
        self.order = torch.randperm(len(datas))
        log(test_log, 'Random Order: {}'.format(self.order))

        sp = 60
        for sess in range(len(datas)):
            data, label = datas[self.order[sess]], labels[self.order[sess]]

            # remap label
            label       = [x.item() for x in label]
            uni_label   = self.get_unique_label(label)
            map_dicts   = dict(zip(uni_label, range(len(uni_label))))
            label       = [map_dicts[x]+sp for x in label] 
            label       = torch.LongTensor(label).cuda()
            sp = sp + 5

            start_label = self.model.base_cls_num + sess * self.model.inc_cls_num
            end_label = self.model.base_cls_num + (sess+1) * self.model.inc_cls_num
            cls_list = np.arange(start_label, end_label)
            self.model      = self.update_inc_fc(self.model, data, label, cls_list)
            self.fsl_model  = self.update_inc_fc(self.fsl_model, data, label, cls_list)

            # get test data
            acc = self.single_session_test_random_order(sess+1)
            self.performance_stati['acc_each_sess'][sess+1] = acc
            log(test_log, 'Sess:{}\tAcc:{:.2f}'.format(sess+1, acc))
        
        log(test_log, 'Performance info: {}'.format(self.performance_stati))
        log(test_log, 'Average: {}'.format(self.performance_stati['acc_each_sess'].mean()))
        return acc


    def single_session_test_cross_domain(self, cur_session: int=0): 

        # testing
        test_accs, collect_preds, collect_labels = [], [], []
        num_seen_cls = self.cross_domain_info['seen_cls']

        # get test data
        for s in range(cur_session+1):
            if s == 0:
                datas   = deepcopy(self.datas_inc_test[str(s)])
                labels  = deepcopy(self.labels_inc_test[str(s)])
            else:
                new_data  = self.datas_inc_test[str(s)]
                new_label = self.labels_inc_test[str(s)]
                for i, data in enumerate(new_data):
                    datas.append(data)
                    labels.append(new_label[i])

        with torch.no_grad():
            for i, data in enumerate(tqdm(datas)):
                label = labels[i]

                x_base      = self.model.encode(data)
                x_fsl       = self.fsl_model.encode(data)

                if self.args.test_method == 'pretrain':
                    proto_base  = self.model.fc.weight[:num_seen_cls]
                    logits_base = F.linear(F.normalize(x_base, p=2, dim=-1), F.normalize(proto_base, p=2, dim=-1)) 
                    scores_     = torch.softmax(logits_base, dim=-1)
                    preds       = torch.argmax(scores_, dim=-1).reshape(-1)

                else:
                    # score given by base
                    proto_base = self.model.fc.weight[:num_seen_cls]
                    logit_base = F.linear(F.normalize(x_base, p=2, dim=-1), F.normalize(proto_base, p=2, dim=-1))
                    # logit_base = self.fsl_model.compute_similarity(proto_base, x_base, mode='cos') wrong because d
        
                    if self.args.output_form_1 == 'softmax':
                        score_base = torch.softmax(logit_base, dim=-1)
                    elif self.args.output_form_1 == 'scaled_softmax':
                        score_base = torch.softmax(self.args.temperature * logit_base, dim=-1)
                    else:
                        score_base = logit_base

                    # score_base = 0

                    # score given by fsl
                    proto_fsl = self.fsl_model.fc.weight[:num_seen_cls]
                    if 'cos' in self.args.sim_metric:
                        logit_fsl = F.linear(F.normalize(x_fsl, p=2, dim=-1), F.normalize(proto_fsl, p=2, dim=-1))
                    else:
                        logit_fsl = self.fsl_model.compute_similarity(proto_fsl, x_fsl, mode=self.args.sim_metric)

                    if self.args.output_form_2 == 'softmax':
                        score_fsl = torch.softmax(logit_fsl, dim=-1)
                    elif self.args.output_form_2 == 'scaled_softmax':
                        score_fsl = torch.softmax(self.args.temperature * logit_fsl, dim=-1)
                    else:
                        score_fsl = logit_fsl

                    # score_fsl = 0

                    # model ensemble
                    score_ens = (score_base + score_fsl) / 2
                    score, preds = torch.max(score_ens, dim=-1)

                accuracy = 100 * preds.eq(label).float().mean()
                test_accs.append(accuracy.item())
                collect_preds.append(preds)
                collect_labels.append(label)

            # performance matrix
            collect_preds = torch.cat(collect_preds, dim=0)
            collect_labels = torch.cat(collect_labels, dim=0)
            if self.model.inc_cls_num != 1:
                self.performance_analysis(cur_session, collect_preds, collect_labels)
            
            # for confusion matrix
            return np.mean(np.array(test_accs))


    def get_inc_test_data(self):
        # incremental train loader
        inc_train_loader = self.get_inc_loader(self.model.inc_cls_num, 
                                               self.inc_shot,
                                               max_cls=self.model.num_cls,
                                               min_cls=self.model.base_cls_num)

        
        for sess, batch in enumerate(tqdm(inc_train_loader)):
            # stage inc train loader
            data, label = [_.cuda() for _ in batch]
            self.datas_inc_train.append(data)
            self.labels_inc_train.append(label)

            # split i test loader
            seen_cls = self.model.base_cls_num + (sess+1) * self.model.inc_cls_num

            # label dicts
            label_dicts = dict(zip(range(seen_cls-self.model.inc_cls_num, seen_cls), 
                                  range(self.cross_domain_info['staged_sp'], 
                                        self.cross_domain_info['staged_sp'] + self.model.inc_cls_num)
                                 ))

            test_loader = self.get_test_loader(seen_cls, seen_cls-self.model.inc_cls_num)
            cls_idx = str(sess + self.cross_domain_info['staged_session'])
            if cls_idx not in self.datas_inc_test:
                self.datas_inc_test[cls_idx] = []
                self.labels_inc_test[cls_idx] = []
            for test_idx, data_label in enumerate(tqdm(test_loader)):
                data, label = [_.cuda() for _ in data_label]

                # remap label
                tmp_label = [x.item() for x in label]
                label = [label_dicts[x] for x in tmp_label]
                label = torch.LongTensor(label).cuda()
                # append dicts
                self.datas_inc_test[cls_idx].append(data)
                self.labels_inc_test[cls_idx].append(label)
            self.cross_domain_info['staged_sp'] += self.model.inc_cls_num


    def test_cross_domain(self, 
                          reload=False, 
                          mode='inc', 
                          base_session: bool=False, 
                          align_first_session:bool=False,
                          cross_domain_set: str='mini_cub_cifar'):
        test_log = os.path.join(self.work_dir, 'test_' + self.args.test_method +'.log')

        if reload:
            self.model = load_trained_paras(self.pre_model_path, [self.model], ['pretrain'])[0]
            if self.args.test_method != 'pretrain':
                fs_model_path   = os.path.join(self.work_dir, self.model_save_name)
                self.fsl_model  = load_trained_paras(fs_model_path, [self.fsl_model], [self.model_save_key])[0]
        self.fsl_model  = self.update_base_fc(self.fsl_model)
        self.model.eval();self.fsl_model.eval()

        # session 0
        acc, _ = self.single_session_test(0)
        self.performance_stati['acc_each_sess'][0] = acc
        log(test_log, 'Sess:{}\tAcc:{:.2f}'.format(0, acc))

        ########################## cross domain
        
        # store data
        self.datas_inc_train  = []
        self.labels_inc_train = []

        # stage test data
        self.datas_inc_test = {}
        self.labels_inc_test = {}

        # split 0
        test_loader = self.get_test_loader(60)
        cls_idx      = str(0)
        if cls_idx not in self.datas_inc_test:
            self.datas_inc_test[cls_idx] = []
            self.labels_inc_test[cls_idx] = []
        for i, batch in enumerate(tqdm(test_loader)):
            data, label = [_.cuda() for _ in batch]
            self.datas_inc_test[cls_idx].append(data)
            self.labels_inc_test[cls_idx].append(label)
        
        self.cross_domain_info = {
            'dataset_base': None,
            'dataset_incs': None,
            'dataset_nums': None,
            'dataset_inc_setting':{
                'mini': [100, 60, 5, 9],
                'cub': [200, 100, 10, 11],
                'cifar': [100, 60, 5, 9],
            },
            'abbr_full':{
                'mini':'miniImageNet',
                'cub': 'cub_200',
                'cifar': 'cifar_fs'
            },
            'staged_session': 1,
            'total_sessions': 1,
            'staged_cls': self.model.base_cls_num,
            'staged_sp': self.model.base_cls_num,
            'seen_cls': self.model.base_cls_num,
            'sess_node': None,

        }

        # split i
        total_sessions = 0
        splits = cross_domain_set.split('_')
        self.cross_domain_info['dataset_base'] = splits[0]
        self.cross_domain_info['dataset_nums'] = len(splits)
        tmp_proto_base = self.model.fc.weight.data[:self.model.base_cls_num]
        tmp_proto_fsl  = self.fsl_model.fc.weight.data[:self.model.base_cls_num]
        if len(splits) > 2: # three datasets
            self.cross_domain_info['dataset_incs'] = [splits[1], splits[2]]
            if splits[0] == 'cub': 
                self.model.fc = nn.Linear(512, 180, bias=False)
                self.fsl_model.fc = nn.Linear(512, 180, bias=False)
                total_sessions = 17

            else:
                self.model.fc = nn.Linear(512, 200, bias=False)
                self.fsl_model.fc = nn.Linear(512, 200, bias=False)
                total_sessions = 19
        else: # two datasets
            self.cross_domain_info['dataset_incs'] = [splits[1]]
            total_sessions = 9
            if splits[0] == 'cub':
                self.model.fc = nn.Linear(512, 140, bias=False)
                self.fsl_model.fc = nn.Linear(512, 140, bias=False)
            if splits[1] == 'cub': # mini_cub or cifar_cub
                self.model.fc = nn.Linear(512, 160, bias=False)
                self.fsl_model.fc = nn.Linear(512, 160, bias=False)
                total_sessions = 11
        self.model.fc.weight.data[:self.model.base_cls_num] = tmp_proto_base
        self.model.fc.cuda()
        self.fsl_model.fc.weight.data[:self.model.base_cls_num] = tmp_proto_fsl
        self.fsl_model.fc.cuda()


        for i, d_name in enumerate(self.cross_domain_info['dataset_incs']):
            self.model.num_cls, self.model.base_cls_num, self.model.inc_cls_num, self.model.sessions = self.cross_domain_info['dataset_inc_setting'][d_name]
            self.args.dataset = self.cross_domain_info['abbr_full'][d_name]
            self.data_prefix = d_name+'_'
            self.get_inc_test_data()
            self.cross_domain_info['staged_cls'] += (self.model.num_cls - self.model.base_cls_num)
            self.cross_domain_info['staged_session'] += self.model.sessions-1

        # 
        self.performance_stati['details'] = torch.zeros(total_sessions, total_sessions)
        self.performance_stati['acc_each_sess'] = torch.zeros(total_sessions)
        self.performance_stati['acc_each_sess'][0] = acc
        #####################

        self.model.num_cls, self.model.base_cls_num, self.model.inc_cls_num, self.model.sessions = self.cross_domain_info['dataset_inc_setting'][splits[1]]
        self.cross_domain_info['sess_node'] = self.model.sessions - 1
        for sess in range(len(self.datas_inc_train)):
            data, label = self.datas_inc_train[sess], self.labels_inc_train[sess]

            if sess+1 > self.cross_domain_info['sess_node']:
                if self.cross_domain_info['dataset_nums'] > 2:
                    self.model.num_cls, self.model.base_cls_num, self.model.inc_cls_num, self.model.sessions = self.cross_domain_info['dataset_inc_setting'][splits[2]]
                    self.cross_domain_info['sess_node'] += (self.model.sessions-1)

            start_label = self.cross_domain_info['seen_cls']
            end_label   = self.cross_domain_info['seen_cls'] + self.model.inc_cls_num
            cls_list    = np.arange(start_label, end_label)
            
            # init fc
            self.model      = self.update_inc_fc(self.model, data, label, cls_list)
            self.fsl_model  = self.update_inc_fc(self.fsl_model, data, label, cls_list)

            # get test data
            self.cross_domain_info['seen_cls'] = end_label
            acc = self.single_session_test_cross_domain(sess+1)

            self.performance_stati['acc_each_sess'][sess+1] = acc
            log(test_log, 'Sess:{}\tAcc:{:.2f}'.format(sess+1, acc))
        
        log(test_log, 'Performance info: {}'.format(self.performance_stati))
        log(test_log, 'Average: {}'.format(self.performance_stati['acc_each_sess'].mean()))
        return acc


    def fsl_test(self, n_w, n_s, n_q, reload=False):
        num_sup = n_w * n_s
        sup_label = torch.arange(n_w).repeat(n_s).reshape(-1).type(torch.cuda.LongTensor)
        que_label = torch.arange(n_w).repeat(n_q).reshape(-1).type(torch.cuda.LongTensor)
        if reload:
            fs_model_path   = os.path.join(self.work_dir, self.model_save_name)
            self.fsl_model  = load_trained_paras(fs_model_path, [self.fsl_model], [self.model_save_key])[0]
        self.fsl_model.eval()
        self.args.state     = 'test'
        self.args.used_data = self.data_prefix + 'test'
        self.args.sampler   = 'fsl'
        fs_val_loader       = get_dataloader(self.args, sample_info=[1000, 1, n_w, n_s, n_q, self.model.base_cls_num])
        val_accs = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(fs_val_loader)):
                data, label         = [_.cuda() for _ in batch]
                sup_data, que_data  = data[:num_sup], data[num_sup:]
                logits = self.fsl_model(sup_data, sup_label, que_data, n_way=n_w, mode=self.args.sim_metric)
                acc    = count_accuracy(logits, que_label)
                val_accs.append(acc.item())
        val_acc_avg = np.mean(np.array(val_accs))
        val_acc_aci95 = 1.96 * np.std(np.array(val_accs)) / np.sqrt(1000)
        return val_acc_avg, val_acc_aci95


    def rotate_image(self, proto_tmp, query_tmp):
        """
        sup_tmp:deep copy from sup_images, shape:[n_way*n_shot, c, h, w]
        que_tmp:deep copy from sup_images, shape:[n_way*n_que, c, h, w]
        """
        for i in range(self.args.n_way):
            # random choose rotate degree
            rot_list = [90, 180, 270]
            sel_rot = random.choice(rot_list)
            if sel_rot == 90:  # rotate 90 degree
                # print('rotate 90 degree')
                proto_tmp[i::self.args.n_way] = proto_tmp[i::self.args.n_way].transpose(2, 3).flip(2)
                query_tmp[i::self.args.n_way] = query_tmp[i::self.args.n_way].transpose(2, 3).flip(2)
            elif sel_rot == 180:  # rotate 180 degree
                # print('rotate 180 degree')
                proto_tmp[i::self.args.n_way] = proto_tmp[i::self.args.n_way].flip(2).flip(3)
                query_tmp[i::self.args.n_way] = query_tmp[i::self.args.n_way].flip(2).flip(3)
            elif sel_rot == 270:  # rotate 270 degree
                # print('rotate 270 degree')
                proto_tmp[i::self.args.n_way] = proto_tmp[i::self.args.n_way].transpose(2, 3).flip(3)
                query_tmp[i::self.args.n_way] = query_tmp[i::self.args.n_way].transpose(2, 3).flip(3)
        return proto_tmp, query_tmp
 

    @torch.no_grad()
    def augment(self, sup_tmp, que_tmp, alpha: float=0.5, mtype: str='inter'):
        n_w, n_s, n_q = self.args.n_way, self.args.n_shot, self.args.n_query
        ori_sup_data = sup_tmp.reshape(n_s, n_w, 3, 224, 224)
        ori_que_data = que_tmp.reshape(n_q, n_w, 3, 224, 224)
        comb = torch.cat((ori_sup_data, ori_que_data), dim=0) # [n_s+n_q, n_w, 3, 224, 224]
        if 'inter' in mtype:
            nums = n_w
        else:
            nums = n_s+n_q
        seq_order = torch.arange(nums)
        gen_order = torch.randperm(nums)
        while 0 in (seq_order-gen_order):
            gen_order = torch.randperm(nums)
            
        if 'inter' in mtype:
            comb = comb.transpose(1, 0) # [n_w, n_s+n_q, 3, 224, 224]
        cand_comb = comb[gen_order]

        betas = torch.distributions.beta.Beta(alpha, alpha).sample([]).item()
        if 'mixup' in mtype:
            syns_comb = betas * comb + (1-betas) * cand_comb

        if 'cutmix' in mtype:
            _, _, _, H, W = ori_sup_data.shape
            rate = np.sqrt(1 - betas)
            cut_x, cut_y = (H * rate) // 2, (W * rate) // 2
            cx, cy = np.random.randint(cut_x, H - cut_x), np.random.randint(cut_y, W - cut_x)
            bx1, bx2 = int(cx - cut_x), int(cx + cut_x)
            by1, by2 = int(cy - cut_y), int(cy + cut_y)
            syns_comb = comb.clone()
            syns_comb[:, :, :, bx1:bx2, by1:by2] = cand_comb[:, :, :, bx1:bx2, by1:by2].clone()

        if 'inter' in mtype:
            syns_comb = syns_comb.transpose(1, 0)

        sup_tmp = syns_comb[:n_s].reshape(-1, 3, 224, 224) # [n_s*n_w, 3, 224, 224]
        que_tmp = syns_comb[n_s:].reshape(-1, 3, 224, 224) # [n_q*n_w, 3, 224, 224]
        return sup_tmp, que_tmp


    @torch.no_grad()
    def update_inc_fc(self, model, data, label, cls_list):
        data    = model.encode(data).detach()
        # proto   = data.reshape(self.model.inc_cls_num, self.inc_shot, -1).mean(dim=1)

        proto   = data.reshape(-1, self.inc_shot, 512).mean(dim=1)
        model.fc.weight.data[cls_list] = proto

        # for class_index in cls_list:
        #     data_index = (label==class_index).nonzero(as_tuple=False).squeeze(-1)
        #     embedding = data[data_index]
        #     proto = embedding.mean(0)
        #     model.fc.weight.data[class_index] = proto
        return model
        

    @torch.no_grad()
    def update_base_fc(self, model):
        base_cls_num = self.model.base_cls_num
        model.eval()
        self.args.state = 'test'
        self.args.used_data = self.data_prefix + 'base'
        self.args.sampler = 'std_official'

        base_loader = get_dataloader(self.args)
        embedding_list = []
        label_list = []
        # data_list=[]
        with torch.no_grad():
            for i, batch in enumerate(tqdm(base_loader)):
                data, label = [_.cuda() for _ in batch]
                embedding = model.encode(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        proto_list = []

        for class_index in range(base_cls_num):
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]
            embedding_this = embedding_this.mean(0)
            proto_list.append(embedding_this)

        proto_list = torch.stack(proto_list, dim=0)
        model.fc.weight.data[:base_cls_num] = proto_list

        return model


    def get_data_of_cm(self, end_label, start_label, topk):
        gt_tmp, pd_tmp, prob_tmp = None, None, None
        self.args.state = 'test'
        self.args.used_data = self.data_prefix+'test'
        self.args.sampler = 'inc_test'
        inc_test_loader = get_dataloader(self.args, 
                            sample_info=[self.args.batch_size_test, end_label, start_label])
        candidate_proto = self.fsl_model.fc.weight
        for i, batch in enumerate(tqdm(inc_test_loader)):
            data, label = [_.cuda() for _ in batch]
            if gt_tmp is None:
                gt_tmp = label.detach().cpu().numpy()
            else:
                gt_tmp = np.hstack((gt_tmp, label.detach().cpu().numpy()))

            x_pre = self.model.encode(data) # [100, 512]
            logits = F.linear(
                F.normalize(x_pre, p=2, dim=-1), F.normalize(self.model.fc.weight, p=2, dim=-1))
            
            pds, probs = None, None
            if self.args.test_method == 'single':
                if self.args.test_model == 'base':
                    probs, pds = acc_topk(
                        logits[:, :end_label], label, k=1, return_ind_p=True)
                elif self.args.test_model == 'fs':
                    query       = self.fsl_model.encode(data) # [n_q, 512]
                    proto       = candidate_proto[:end_label, :] # [n, 512]
                    fsl_logits  = self.fsl_model.compute_similarity(proto, query, mode='euc')
                    probs, pds  = acc_topk(
                        fsl_logits[:, :end_label], label, k=1, return_ind_p=True)
                else:
                    AssertionError("Invalid model name")
                probs = probs.view(-1)
                pds = pds.view(-1)

            elif self.args.test_method == 'db':
                pre_values, pre_ind = acc_topk(
                                logits[:, :end_label], label, k=topk, return_ind_p=True)
                fs_que_features = self.fsl_model.encode(data) # [100, 512]
                preds = []
                n, c, h, w = data.size()
                # fsl_scores = []
                
                for ind in range(pre_ind.shape[0]): 
                    topk_ind = pre_ind[ind].view(-1)
                    topk_val = pre_values[ind].reshape(1, topk)
                    select_proto = candidate_proto[topk_ind] # [topk, 512]
                    cur_que_feature = fs_que_features[ind].reshape(1, fs_que_features.shape[-1]) # [1, 512]
                    fsl_logits = self.fsl_model.compute_similarity(select_proto, cur_que_feature, mode='euc')
                    fsl_logits = torch.softmax(fsl_logits, dim=-1)
                    cal_logits = fsl_logits * topk_val
                    # cal_logits = 1*fsl_logits + topk_val
                    cal_ind = torch.argmax(cal_logits, dim=-1)
                    pre = topk_ind[cal_ind]
                    val = topk_val.view(-1)[cal_ind]
                    if pds is None:
                        pds = pre
                        probs = val
                    else:
                        pds = torch.cat((pds, pre), 0)
                        probs = torch.cat((probs, val), 0)
            else:
                AssertionError("Invalid test method")
        
            if pd_tmp is None:
                pd_tmp = pds.detach().cpu().numpy()
                prob_tmp = probs.detach().cpu().numpy()
            else:
                pd_tmp = np.hstack((pd_tmp, pds.detach().cpu().numpy()))
                prob_tmp = np.hstack((prob_tmp, probs.detach().cpu().numpy()))
        return gt_tmp, pd_tmp, prob_tmp


    def get_confusion_matrix(self):
        import numpy as np
        topk = self.args.topk
        fs_model_path = os.path.join(self.work_dir, self.model_save_name)
        self.fsl_model = load_trained_paras(fs_model_path, [self.fsl_model], [self.model_save_key])[0]
        self.model = load_trained_paras(self.pre_model_path, [self.model], ['pretrain'])[0]
        self.model = self.update_base_fc(self.model)
        self.fsl_model = self.update_base_fc(self.fsl_model)
        self.model = deepcopy(self.model)
        self.fsl_model = deepcopy(self.fsl_model)
        # incremental train loader
        self.args.state = 'test'
        self.args.used_data = self.data_prefix + 'inc'
        self.args.sampler = 'inc'
        inc_train_loader = get_dataloader(self.args, sample_info=[self.model.inc_cls_num, 5])
        self.model.eval()
        self.fsl_model.eval()
        gt, pd, prob = None, None, None
        with torch.no_grad():
            for sess, batch in enumerate(tqdm(inc_train_loader)):
                data, label = [_.cuda() for _ in batch]
                start_label = self.model.base_cls_num + sess * self.model.inc_cls_num
                end_label = self.model.base_cls_num + (sess+1) * self.model.inc_cls_num
                cls_list = np.arange(start_label, end_label)
                self.model = self.update_inc_fc(self.model, data, label, cls_list)
                self.fsl_model = self.update_inc_fc(self.fsl_model, data, label, cls_list)

                if sess == 0:
                    gt_tmp, pd_tmp, prob_tmp = self.get_data_of_cm(
                                        self.model, self.fsl_model, start_label, 0, topk)
                    gt, pd, prob = gt_tmp, pd_tmp, prob_tmp 
                gt_tmp, pd_tmp, prob_tmp = self.get_data_of_cm(
                                    self.model, self.fsl_model, end_label, start_label, topk)
                gt = np.hstack((gt, gt_tmp))
                pd = np.hstack((pd, pd_tmp))
                prob = np.hstack((prob, prob_tmp))

                
        np.savez('L2I_high_acc.npz', pcls=pd, tcls=gt, prob=prob)


    def get_data_for_TSNE(self):
        import numpy as np
        fs_model_path = os.path.join(self.work_dir, self.model_save_name)
        self.fsl_model = load_trained_paras(fs_model_path, [self.fsl_model], [self.model_save_key])[0]
        self.model = load_trained_paras(self.pre_model_path, [self.model], ['pretrain'])[0]
        self.model = self.update_base_fc(self.model)
        self.fsl_model = self.update_base_fc(self.fsl_model)
        self.model = deepcopy(self.model)
        self.fsl_model = deepcopy(self.fsl_model)
        # incremental train loader
        self.args.state = 'test'
        self.args.used_data = self.data_prefix + 'inc'
        self.args.sampler = 'inc'
        inc_train_loader = get_dataloader(self.args, sample_info=[self.model.inc_cls_num, 5])
        self.model.eval()
        self.fsl_model.eval()

        with torch.no_grad():
            for sess, batch in enumerate(tqdm(inc_train_loader)):
                data, label = [_.cuda() for _ in batch]
                start_label = self.model.base_cls_num + sess * self.model.inc_cls_num
                end_label = self.model.base_cls_num + (sess+1) * self.model.inc_cls_num
                cls_list = np.arange(start_label, end_label)
                self.model = self.update_inc_fc(self.model, data, label, cls_list)
                self.fsl_model = self.update_inc_fc(self.fsl_model, data, label, cls_list)
        
            feature_base, feature_fs, label_all = None, None, None

            self.args.state = 'test'
            self.args.used_data = self.data_prefix+'test'
            self.args.sampler = 'inc_test'
            inc_test_loader = get_dataloader(self.args, 
                                sample_info=[self.args.batch_size_test, self.model.num_cls])
            for i, batch in enumerate(tqdm(inc_test_loader)):
                data, label = [_.cuda() for _ in batch]
                bs_feature = self.model.encode(data)
                fs_feature = self.fsl_model.encode(data)
                if feature_base is None:
                    feature_base = bs_feature
                    feature_fs = fs_feature
                    label_all = label.view(-1)
                else:
                    feature_base = torch.cat((feature_base, bs_feature), 0)
                    feature_fs = torch.cat((feature_fs, fs_feature), 0)
                    label_all = torch.cat((label_all, label.view(-1)), 0)
            
            protos_base = self.model.fc.weight.detach().cpu().numpy()
            protos_fs = self.fsl_model.fc.weight.detach().cpu().numpy()
            feature_base = feature_base.detach().cpu().numpy()
            feature_fs = feature_fs.detach().cpu().numpy()
            label_all = label_all.detach().cpu().numpy()
            np.savez('L2I_data_for_TSNE_highAcc.npz', 
                    proto_base=protos_base, protos_fs=protos_fs, feature_base=feature_base, feature_fs=feature_fs,
                    label=label_all)

      
    def init_paths(self):
        # 
        self.pre_model_path = None
        if self.args.dataset == 'miniImageNet':
            # examplars
            self.examplar_num_per_cls = 20
            self.data_prefix = 'mini_'
            self.pretrain_dir = os.path.join(self.cwd, 'experiments/pretrain/mini/pretrain_mini')
            self.work_dir = os.path.join(self.cwd, 'experiments/RCN/mini/'+self.args.storage_folder)  # 

            if self.args.pretrained_model == 'yz_mxup':
                print("use model pretrained with mixup by yz")
                self.pre_model_path  = 'experiments/pretrain/mini/pretrain_mini_mixup_yz/session0.pth'
            elif self.args.pretrained_model == 'wy_mxup':
                print("use model pretrained with mixup by wy")
                self.pre_model_path  = 'experiments/pretrain/mini/pretrain_mini_mixup/session0.pth'
            elif self.args.pretrained_model == 'std_pretrain':
                print("use std pretrained model")
                self.pre_model_path  = 'experiments/pretrain/mini/pretrain_mini/session0.pth'
            else:
                self.args.reload=False
                print("Do not use any pretrained model!")
        elif self.args.dataset == 'cifar_fs':
            # examplars
            self.examplar_num_per_cls = 20
            self.data_prefix = 'cifar_'
            self.pretrain_dir = os.path.join(self.cwd, 'experiments/pretrain/cifar/pretrain_cifar')
            self.work_dir = os.path.join(self.cwd, 'experiments/RCN/cifar/'+self.args.storage_folder)  # 

            if self.args.pretrained_model == 'yz_mxup':
                print("use model pretrained with mixup by yz")
                self.pre_model_path = 'experiments/pretrain/cifar/pretrain_cifar_mixup_yz/session0.pth'

            elif self.args.pretrained_model == 'wy_mxup':
                print("use model pretrained with mixup by wy")
                self.pre_model_path = 'experiments/pretrain/cifar/pretrain_cifar_mixup/session0.pth'

            elif self.args.pretrained_model == 'std_pretrain':
                print("use std pretrained model") 
                self.pre_model_path = 'experiments/pretrain/cifar/pretrain_cifar/session0.pth'
            else:
                self.args.reload=False
                print("Do not use any pretrained model!")
        elif self.args.dataset == 'cub_200':
            # examplars
            self.examplar_num_per_cls = 5
            self.data_prefix = 'cub_'
            self.pretrain_dir = os.path.join(self.cwd, 'experiments/pretrain/cub/pretrain_cub')
            self.work_dir = os.path.join(self.cwd, 'experiments/RCN/cub/'+self.args.storage_folder)  # 
            # yz mixup pretrain
            if self.args.pretrained_model == 'yz_mxup':
                print("use model pretrained with mixup by yz")
                self.pre_model_path = 'experiments/pretrain/cub/pretrain_cub_mixup_yz/session0.pth'
            elif self.args.pretrained_model == 'wy_mxup':
                print("use model pretrained with mixup by wy")
                # our mixup pretrain
                self.pre_model_path = 'experiments/pretrain/cub/pretrain_cub_mixup/session0.pth'
            elif self.args.pretrained_model == 'std_pretrain':
                print("use std pretrained model")
                # w/o mixup pretrain
                self.pre_model_path = 'experiments/pretrain/cub/pretrain_cub/session0.pth'
            else:
                self.args.reload=False
                print("Do not use any pretrained model!")
        else:
            raise Exception("invalid dataset name in CFTrainer")
        # 
        if self.pre_model_path is not None:
            self.pre_model_path = os.path.join(self.cwd, self.pre_model_path)
            print(f"The path of pretrained model is {self.pre_model_path}")
        

    def performance_analysis(self, cur_session: int, collect_preds: torch.tensor, collect_labels: torch.tensor):
        for i in range(cur_session+1):
            if i == 0:
                start_ = 0
                end_   = self.model.base_cls_num
            else:
                start_ = self.model.base_cls_num + (i-1) * self.model.inc_cls_num
                end_   = self.model.base_cls_num + i * self.model.inc_cls_num
            
            # label belong to [start_, end_)
            idx_lt = torch.lt(collect_labels, end_).nonzero(as_tuple=False).squeeze()
            idx_ge = torch.ge(collect_labels, start_).nonzero(as_tuple=False).squeeze()
            idx    = [j for j in idx_ge if j in idx_lt]

            select_preds  = collect_preds[idx]
            select_labels = collect_labels[idx]

            # calc acc
            cur_acc = 100 * select_preds.eq(select_labels).float().mean()
            self.performance_stati['details'][i][cur_session] = cur_acc
        
        # last session
        if cur_session == self.model.sessions - 1:
            details                 = self.performance_stati['details']
            # forgetting
            performance_prev        = details[:-1, :-1] # k-1 step
            performance_prev_max, _ = torch.max(performance_prev, dim=-1)
            performance_last        = details[:-1, -1]  # k step
            forgetting              = torch.mean(performance_prev_max-performance_last).item()

            # acc base and acc novel
            last_base               = details[0, -1].item()
            acc_novel               = torch.mean(details[1:, -1]).item()

            self.performance_stati['forgetting']    = forgetting
            self.performance_stati['acc_base']      = last_base
            self.performance_stati['acc_novel']     = acc_novel


    # pretrain relevant
    @torch.no_grad()
    def joint_test(self, reload: bool=False, cur_session: int=0):
        # load model if necessary
        if reload:
            self.model = load_trained_paras(os.path.join(self.work_dir, 'session0.pth'), 
                                            [self.model], ['pretrain'])[0]

        # initial
        test_log = os.path.join(self.work_dir, 'test_' + str(cur_session) + '.log')
        num_seen_cls = self.model.base_cls_num + cur_session * self.model.inc_cls_num
        test_accs, collect_preds, collect_labels = [], [], []

        # prepare data
        inc_test_loader = self.get_test_loader(num_seen_cls)

        self.model.eval()
        for i, batch in enumerate(tqdm(inc_test_loader)):
            # get dara
            data, label = [_.cuda() for _ in batch]

            # get logits
            x = self.model.encode(data)
            logits = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.model.fc.weight, p=2, dim=-1))

            # predictions
            preds       = torch.argmax(torch.softmax(logits[:, :num_seen_cls], dim=-1), dim=-1).reshape(-1)
            accuracy    = 100 * preds.eq(label).float().mean()
            test_accs.append(accuracy.item())

            collect_preds.append(preds)
            collect_labels.append(label)
        collect_preds   = torch.cat(collect_preds, dim=0)
        collect_labels  = torch.cat(collect_labels, dim=0)

        # pdb.set_trace()
        # if cur_session == 10:
        #     pds = collect_preds.cpu().numpy()
        #     gts = collect_labels.cpu().numpy()
        #     np.savez('PlotFactory/REPO/JointCNN.npz', pds=pds, gts=gts)

        self.performance_analysis(cur_session, collect_preds, collect_labels)
        return np.mean(np.array(test_accs))


    def pretrain_process(self, train_log_path, model_save_path, joint: bool=False, cur_session: int=0):
        num_seen_cls = self.model.base_cls_num + cur_session * self.model.inc_cls_num

        # set optimizer
        optimizer, scheduler =  get_optimizer(self.model, self.args)

        # base train loader
        train_loader = self.get_train_loader('std', 
                                             joint=joint, 
                                             cur_session=cur_session,
                                             full_data=False)

        # training 
        max_acc = 0
        timer = Timer()
        for epoch in range(1, self.args.epoch+1):
            log(train_log_path, 'Session:{}\tTrain Epoch:{}\tLearning Rate:{:.6f}'.format(cur_session, epoch, scheduler.get_last_lr()[0]))
            self.model.train()
            train_losses = []
            for i, batch in enumerate(tqdm(train_loader)):
                # get data
                data, label = [_.cuda() for _ in batch]

                # get logits
                x = self.model.encode(data)
                logits = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.model.fc.weight, p=2, dim=-1))

                # get loss
                loss = F.cross_entropy(self.args.temperature * logits[:,:num_seen_cls], label)

                # optimize the model
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                train_losses.append(loss.item())

                if i % 20 == 0:
                    loss_avg = np.mean(np.array(train_losses))
                    log(train_log_path, 'Session:{}\t Epoch:{}\tBatch:[{}/{}]\tLoss:{:.4f} % ({:.4f})\t'.format(
                        cur_session, epoch, i, len(train_loader), loss_avg, loss))
            scheduler.step()

            if epoch % 10 == 0:
                # testing phase
                if joint:
                    test_acc = self.joint_test(cur_session=cur_session)
                else:
                    test_acc = self.test()

                if test_acc > max_acc:
                    self.performance_stati['acc_each_sess'][cur_session] = test_acc
                    max_acc = test_acc
                    torch.save({'pretrain': self.model.state_dict()}, model_save_path)
                log(train_log_path, "Elapsed Time: {}/{}\tacc:{:.2f}\n".format(
                    timer.measure(), timer.measure(epoch/float(self.args.epoch)), test_acc))

    # pretraining using standard classification training paradigm
    def pretraining(self, mode='cos', reload=False, joint:bool=False):

        if joint:
            print("Joint-training......")
            for sess in range(self.model.sessions):
                train_log_path = os.path.join(self.work_dir, f'train{sess}.log')
                log(train_log_path, str(vars(self.args)), mode='w+')
                model_save_path = os.path.join(self.work_dir, f'session{sess}.pth')
                self.pretrain_process(train_log_path, model_save_path, joint=joint, cur_session=sess)
                # reinitialize model
                # self.model = MyModel(dataset=self.args.dataset,
                #                      arch_name=self.args.network,
                #                      prompt_len=self.args.prompt_len,
                #                      run_method=self.args.run_method).to(self.device)
                # self.model = load_trained_paras(model_save_path, [self.model], ['pretrain'])[0]
                log(train_log_path, 'Performance info:{}'.format(self.performance_stati))
            log(train_log_path, 'acc_in_each_sess when joint-training: {}'.format(self.performance_stati['acc_each_sess']))
        else:
            print("Pretraining......")
            train_log_path = os.path.join(self.work_dir, 'train.log')
            log(train_log_path, str(vars(self.args)), mode='w+')
            model_save_path = os.path.join(self.work_dir, 'session0.pth')
            self.pretrain_process(train_log_path, model_save_path)
  
