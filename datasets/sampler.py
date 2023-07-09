import numpy as np
import torch
from torch.utils.data import Sampler
import torch.distributed as dist
import math
from copy import deepcopy

__all__ = ['StandardSampler', 'inc_sampler', 'Batch_Sample_From_Range', 'Auto_Task_Sampler', 'Cub_Test_Sampler']

# e.g. 
class Cub_Test_Sampler(Sampler):
    # label of class start from 0 to (maxcls-1)
    def __init__(self, label, batch_size, max_cls=10000, min_cls=0):
        self.batch_size = batch_size
        self.inds = {}
        for i, class_id in enumerate(label):
            if class_id >= min_cls and class_id < max_cls:
                if class_id not in self.inds:
                    self.inds[class_id] = []
                self.inds[class_id].append(i)
    
    def __len__(self):
        return len(self.inds.keys())

    def __iter__(self):
        temp_inds = deepcopy(self.inds)
        # for batch_num in range(len(self.inds.keys())):
        for k in temp_inds.keys():
            id_list = temp_inds[k]
            # id_list = temp_inds[batch_num+100] # delete later
            yield id_list
# e.g. 
class Batch_Sample_From_Range(Sampler):
    # label of class start from 0 to (maxcls-1)
    def __init__(self, label, batch_size, max_cls=10000, min_cls = 0):
        self.batch_size = batch_size
        self.inds = []
        for i, class_id in enumerate(label):
            if class_id >= min_cls and class_id < max_cls:
                self.inds.append(i)
    
    def __len__(self):
        return math.ceil(len(self.inds) // self.batch_size)

    def __iter__(self):
        temp_inds = deepcopy(self.inds)
        np.random.shuffle(temp_inds)
        batches = math.ceil(len(self.inds) // self.batch_size)
        for batch_num in range(batches):
            if (batch_num + 1) * self.batch_size <= len(temp_inds):
                id_list = temp_inds[batch_num * self.batch_size : (batch_num + 1) * self.batch_size]
            else:
                id_list = temp_inds[batch_num * self.batch_size : len(temp_inds)]
            yield id_list


class inc_sampler(Sampler):
    def __init__(self, 
                 label,
                 n_way,  # num of cls in each task
                 n_shot,  # num of samples in each cls used to get proto
                 max_cls=10000, 
                 min_cls=0
                 ):
        self.way = n_way
        self.shot = n_shot

        class2id = {}
        for i, class_id in enumerate(label):
            if class_id >= min_cls and class_id < max_cls:
                if class_id not in class2id:
                    class2id[class_id] = []
                class2id[class_id].append(i)
        
        self.class2id = {}
        sort_keys = sorted(class2id)
        for k in sort_keys:
            self.class2id[k] = class2id[k]

    def __len__(self):
        return len(self.class2id) // self.way

    def __iter__(self):

        temp_class2id = deepcopy(self.class2id)
        for class_id in temp_class2id:
            np.random.shuffle(temp_class2id[class_id])

        while len(temp_class2id) >= self.way:
            id_list = []

            list_class_id = list(temp_class2id.keys())
            batch_class_id = list_class_id[:self.way]
            for class_id in batch_class_id:
                if len(temp_class2id[class_id]) < self.shot:
                    valid_nums = len(temp_class2id[class_id])
                else:
                    valid_nums = self.shot
                for _ in range(valid_nums):
                    id_list.append(temp_class2id[class_id].pop())
                

            for class_id in batch_class_id:
                temp_class2id.pop(class_id)
            yield id_list

class Task_Sample_From_Range(Sampler):
    def __init__(self, 
                 label,
                 n_way,  # num of cls in each task
                 n_shot,  # num of samples in each cls used to get proto
                 max_cls=10000,
                 min_cls=0
                 ):
        self.way = n_way
        self.shot = n_shot

        class2id = {}

        for i, class_id in enumerate(label):
            if class_id >= min_cls and  class_id < max_cls:
                if class_id not in class2id:
                    class2id[class_id] = []
                class2id[class_id].append(i)

        self.class2id = class2id
    
    def __len__(self):
        return len(self.class2id) // self.way

    def __iter__(self):

        temp_class2id = deepcopy(self.class2id)
        for class_id in temp_class2id:
            np.random.shuffle(temp_class2id[class_id])

        while len(temp_class2id) >= self.way:

            id_list = []

            list_class_id = list(temp_class2id.keys())
            batch_class_id = list_class_id[:self.way]
            for class_id in batch_class_id:
                for _ in range(self.shot):
                    id_list.append(temp_class2id[class_id].pop())

            for class_id in batch_class_id:
                temp_class2id.pop(class_id)
            yield id_list

class StandardSampler(Sampler):
    def __init__(self,
                 label,
                 batch_size,  
                 state,
                 drop_last = True,
                 train_sample_each_c=550
                 ):
        self.bs = batch_size
        self.drop_last = drop_last
        label = np.array(label)
        self.m_ind = []
        unique = np.sort(np.unique(label))
        for i in unique:
            ind = np.argwhere(label == i).reshape(-1)
            lens = ind.shape[0]
            if train_sample_each_c < 1:
                train_sample_each_c = int(train_sample_each_c * lens)
            if state == 'train':
                ind = torch.from_numpy(ind[:train_sample_each_c])
            elif state == 'val':
                ind = torch.from_numpy(ind[train_sample_each_c:])
            elif state == 'test' :
                ind = torch.from_numpy(ind)
            else:
                raise Exception("Error occur in Sampler_Normal!")
            self.m_ind.append(ind) 
        self.m_ind = torch.cat(self.m_ind, dim=-1).t()

        self.batches = int(self.m_ind.shape[0] / self.bs)
        if self.m_ind.shape[0] - self.batches * self.bs <  self.bs:
            if not self.drop_last:
                self.batches = self.batches + 1

    def __len__(self):
        return self.batches
        
    def __iter__(self):
        index = torch.randperm(self.m_ind.shape[0])
        for i in range(self.batches):
            start = int(i*self.bs)
            end = int((i+1)*self.bs)
            if end < self.m_ind.shape[0]+1:
                batch = self.m_ind[index[start:end]]
                yield batch
            else:
                if not self.drop_last:
                    batch = self.m_ind[index[start:]]
                    yield batch


# sample n_iter tasks: int iter loop, each task may have different cls in each sample procedure
class CategoriesSampler(Sampler):
    def __init__(self,
                 label,
                 batches,  # num of batches we will go to sample
                 task_per_batch,  # num of tasks in each batch,val,test=1
                 n_way,  # num of cls in each task
                 n_shot,  # num of samples in each cls used to get proto
                 n_query,  # num of samples in each cls used to get result
                 max_cls = 200, 
                 min_cls = 0,
                 cls_list=None
                 ):
        self.task_per_batch = task_per_batch
        self.batches = batches
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.cls_list = cls_list
        label = np.array(label)
        self.m_ind = []
        unique = np.sort(np.unique(label))  # delete rep ele, then rank by ascent order
        for i in unique:
            if self.cls_list is None:
                if i >= min_cls and i < max_cls:
                    ind = np.argwhere(label == i).reshape(-1)
                    ind = torch.from_numpy(ind)
                    self.m_ind.append(ind) 
            else:
                if i in self.cls_list:
                    ind = np.argwhere(label == i).reshape(-1)
                    ind = torch.from_numpy(ind)
                    self.m_ind.append(ind) 
        

    def __len__(self):
        return self.batches
        
    # batch->task->n_way,k_shot
    def __iter__(self):
        for i in range(self.batches):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_way]  #
            for j in range(self.task_per_batch):
                for c in classes:
                    l = self.m_ind[c.item()]
                    pos = torch.randperm(l.size()[0])
                    batch.append(l[pos[:self.n_shot + self.n_query]])  # include each label's position
            batch = torch.stack(batch).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch
      
      
# sampler used for meta-training
class Auto_Task_Sampler(Sampler):
    def __init__(self, 
                 label,
                 n_way,  # num of cls in each task
                 n_shot,  # num of samples in each cls used to get proto
                 n_query,  # num of samples in each cls used to get result):
                 cls_list=None
                 ):
        self.data_nums = 0
        self.way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        if n_query != 0:
            self.shots = [self.n_shot, self.n_query]
        else:
            self.shots = [self.n_shot]

        class2id = {}
        for i, class_id in enumerate(label):
            if cls_list is None:
                if class_id not in class2id:
                    class2id[class_id] = []
                class2id[class_id].append(i)
                self.data_nums += 1
            else:
                if class_id in cls_list:
                    if class_id not in class2id:
                        class2id[class_id] = []
                    class2id[class_id].append(i)
                    self.data_nums += 1

        self.class2id = class2id
    
    def __len__(self):
        return self.data_nums // (self.way*(self.n_shot+self.n_query))

    def __iter__(self):

        temp_class2id = deepcopy(self.class2id)
        for class_id in temp_class2id:
            np.random.shuffle(temp_class2id[class_id])

        while len(temp_class2id) >= self.way:

            id_list = []

            list_class_id = list(temp_class2id.keys())

            pcount = np.array([len(temp_class2id[class_id]) for class_id in list_class_id])

            batch_class_id = np.random.choice(list_class_id, size=self.way, replace=False, p=pcount / sum(pcount))

            for shot in self.shots:
                for class_id in batch_class_id:
                    for _ in range(shot):
                        id_list.append(temp_class2id[class_id].pop())

            for class_id in batch_class_id:
                if len(temp_class2id[class_id]) < sum(self.shots):
                    temp_class2id.pop(class_id)

            yield id_list



# sample n_iter tasks: int iter loop, each task may have different cls in each sample procedure
class CategoriesSamplerAlignInc(Sampler):
    def __init__(self,
                 label,
                 batches,  # num of batches we will go to sample
                 task_per_batch,  # num of tasks in each batch,val,test=1
                 n_way,  # num of cls in each task
                 n_shot,  # num of samples in each cls used to get proto
                 n_query,  # num of samples in each cls used to get result
                 max_cls = 200, 
                 min_cls = 0,
                 batch_size = 128,
                 cls_list=None
                 ):
        self.task_per_batch = task_per_batch
        self.batches = batches
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.cls_list = cls_list
        self.batch_size = batch_size
        label = np.array(label)
        self.m_ind = []
        unique = np.sort(np.unique(label))  # delete rep ele, then rank by ascent order
        for i in unique:
            if self.cls_list is None:
                if i >= min_cls and i < max_cls:
                    ind = np.argwhere(label == i).reshape(-1)
                    ind = torch.from_numpy(ind)
                    self.m_ind.append(ind) 
            else:
                if i in self.cls_list:
                    ind = np.argwhere(label == i).reshape(-1)
                    ind = torch.from_numpy(ind)
                    self.m_ind.append(ind) 
        

    def __len__(self):
        return self.batches
        
    # batch->task->n_way,k_shot
    def __iter__(self):
        for i in range(self.batches):
            batch = []
            classes      = torch.randperm(len(self.m_ind))
            inc_classes  = classes[:self.n_way]  #
            base_classes = classes[self.n_way:]  #
            for j in range(self.task_per_batch):
                for c in inc_classes:
                    l = self.m_ind[c.item()]
                    pos = torch.randperm(l.size()[0])
                    batch.append(l[pos[:self.n_shot + self.n_query]])  # include each label's position
            batch = torch.stack(batch).t().reshape(-1)

            # sample base data
            base_data = []
            for c in base_classes:
                tmp_data = self.m_ind[c.item()]
                base_data.append(tmp_data)
            # base_data  = torch.stack(base_data).reshape(-1)
            base_data  = torch.cat(base_data, dim=-1).reshape(-1)
            batch_base = base_data[torch.randperm(base_data.shape[0])[:self.batch_size]]
            batch = torch.cat((batch, batch_base), dim=0)
            
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch

# if __name__ == '__main__':
#     label = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 9, 9]
#     print(len(label))
#     sample = Sampler_Normal(label, 2, 'val', 2)
#     batches = sample.__iter__()
#     for b in batches:
#         print(b)