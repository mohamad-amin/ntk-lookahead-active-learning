import os
import sys
import pickle
from copy import deepcopy
from collections import OrderedDict
import numpy as np
from scipy import stats
import torch
from torch import nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import Event, Queue
from torch.autograd import Variable
from sklearn.metrics import pairwise_distances

from src.utils import util
from src.builders import optimizer_builder, scheduler_builder, criterion_builder


def compute_margin(probs):
    probs = normalize_probs(probs)
    argsort = torch.argsort(probs, dim=1)[-2:]
    probs = probs[torch.arange(probs.shape[0]), argsort]
    margin = probs[:,1] - probs[:,0]
    return margin


def normalize_probs(probs):  # The transposes are for broadcasting over the first dimension
    probs = (probs - probs.min(dim=1)[0].unsqueeze(-1)) / (probs.max(dim=1)[0] - probs.min(dim=1)[0]).unsqueeze(-1)
    probs = torch.where(probs == 0, 1e-6 * torch.ones_like(probs).to(probs.device), probs)
    sum_of_probs = probs.sum(dim=1, keepdims=True)
    probs = probs / sum_of_probs
    return probs


def compute_entropy(probs):
    probs = normalize_probs(probs)
    probs = torch.where(probs == 0, 1e-2 * torch.ones_like(probs).to(probs.device), probs)
    entropy = -1.0 * torch.sum(probs * torch.log(probs), axis=1)
    return entropy


def compute_l2_risk(probs):
    ones = torch.ones_like(probs)
    indices = torch.argmin(torch.abs(ones - probs), dim=1)
    unbounded_probs = probs[torch.arange(indices.shape[0]), indices]
    bounded_probs = torch.where(unbounded_probs >= 1., torch.ones_like(unbounded_probs), unbounded_probs)
    risk = torch.sum(1 - bounded_probs)
    return risk


def init_process(gpu, dataloaders, models, model_config, train_config, data_config, logger, save_dir):
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(gpu)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8888'
    print("Use GPU: {} for training".format(gpu))
    dist.init_process_group(backend='nccl',
                            #init_method='127.0.0.1:8888',
                            world_size=num_gpus,
                            rank=gpu)

    batch_size = data_config['batch_size']
    num_workers = data_config['num_workers']

    labeled_dataset, unlabeled_dataset =\
        dataloaders['train'].dataset, dataloaders['unlabeled'].dataset
    labeled_set = dataloaders['labeled_set']
    subset = dataloaders['subset']
    unlabeled_dataloader = dataloaders['unlabeled']
    num_classes = model_config['model_arch']['num_classes']

    total_exp_entropy = []
    for i, unlabeled_idx in enumerate(subset):
        if i % num_gpus != gpu:
            continue
        # Build a new train dataloader
        new_labeled_set = deepcopy(labeled_set)
        new_labeled_set.append(unlabeled_idx)
        new_train_dataloader = DataLoader(labeled_dataset, batch_size=batch_size,
                                sampler=SubsetRandomSampler(new_labeled_set),
                                num_workers=num_workers)

        # Load the retrain model
        retrain_model = DDP(models['retrain_model'].to(gpu), device_ids=[gpu])
        checkpoint = models['model'].state_dict()
        retrain_model.load_state_dict(checkpoint, strict=True)

        # Build an optimizer and criterion
        optimizer = optimizer_builder.build(
            train_config['optimizer'], retrain_model.parameters(), logger)
        criterion = criterion_builder.build(train_config, model_config, logger)

        # Conpute inductive probs
        cand_input_dict = unlabeled_dataset[unlabeled_idx]
        cand_input_dict['inputs'] = cand_input_dict['inputs'].unsqueeze(0)
        cand_input_dict = util.to_device(cand_input_dict, gpu)
        output_dict = models['model'](cand_input_dict)
        induced_probs = torch.softmax(output_dict['logits'], dim=1)[0].to(gpu)
        exp_entropy = 0.
        for c in range(num_classes):
            labels = torch.tensor([c]).long().to(gpu)
            retrain_model.train()
            for _ in range(max(int(train_config['num_epochs'] / 10), 5)):
                for input_dict in new_train_dataloader:
                    input_dict = util.to_device(input_dict, gpu)

                    optimizer.zero_grad()
                    output_dict = retrain_model(input_dict)
                    output_dict['labels'] = input_dict['labels']

                    indices = input_dict['indices']
                    idx = torch.where(indices == unlabeled_idx)[0]
                    if len(idx) > 0:
                        output_dict['labels'][idx] = labels

                    losses = criterion(output_dict)
                    loss = losses['loss']
                    loss.backward()
                    optimizer.step()

            retrain_model.eval()
            with torch.no_grad():
                entropy = 0.
                for j, unlabeled_input_dict in enumerate(unlabeled_dataloader):
                    if i == j: continue
                    unlabeled_input_dict = util.to_device(unlabeled_input_dict, gpu)

                    output_dict = retrain_model(unlabeled_input_dict)
                    probs = torch.softmax(output_dict['logits'], dim=1)
                    entropy += (-torch.sum(probs * torch.log(probs)))

                exp_entropy += induced_probs[c] * entropy

        with open(os.path.join(save_dir, '{}.pickle'.format(i)), 'wb') as f:
            pickle.dump(-1.0 * exp_entropy.item(), f)
        if gpu == 0:
            print('Retraining is done - {}/{}'.format(i, len(subset)))


def get_eer_parallel(dataloaders, models, model_config, train_config, data_config, logger, device, save_dir):
    save_dir = os.path.join(save_dir, 'uncertainty')
    os.makedirs(save_dir, exist_ok=True)
    mp.spawn(init_process, nprocs=torch.cuda.device_count(),
                 args=(dataloaders, models, model_config, train_config, data_config, None, save_dir))

    total_exp_entropy = []
    for i in range(len(dataloaders['subset'])):
        with open(os.path.join(save_dir, '{}.pickle'.format(i)), 'rb') as f:
            exp_entropy = pickle.load(f)
            total_exp_entropy.append(exp_entropy)

    return total_exp_entropy


def get_eer(dataloaders, models, model_config, train_config, data_config, logger, device, save_dir):
    batch_size = data_config['batch_size']
    num_workers = data_config['num_workers']

    labeled_dataset, unlabeled_dataset =\
        dataloaders['train'].dataset, dataloaders['unlabeled'].dataset
    labeled_set = dataloaders['labeled_set']
    subset = dataloaders['subset']
    unlabeled_dataloader = dataloaders['unlabeled']
    num_classes = model_config['model_arch']['num_classes']

    total_exp_entropy = []
    for i, unlabeled_idx in enumerate(subset):
        # Build a new train dataloader
        new_labeled_set = deepcopy(labeled_set)
        new_labeled_set.append(unlabeled_idx)
        new_train_dataloader = DataLoader(labeled_dataset, batch_size=batch_size,
                                sampler=SubsetRandomSampler(new_labeled_set),
                                num_workers=num_workers)

        # Load the retrain model
        retrain_model = models['retrain_model']
        if torch.cuda.device_count() > 1:
            retrain_model = util.DataParallel(retrain_model)
        retrain_model.to(device)

        checkpoint = models['model'].state_dict()
        retrain_model.load_state_dict(checkpoint, strict=True)

        # Build an optimizer and criterion
        optimizer = optimizer_builder.build(
            train_config['optimizer'], retrain_model.parameters(), logger)
        criterion = criterion_builder.build(train_config, model_config, logger)

        # Conpute inductive probs
        input_dict = unlabeled_dataset[unlabeled_idx]
        input_dict['inputs'] = input_dict['inputs'].unsqueeze(0)
        input_dict = util.to_device(input_dict, device)
        output_dict = models['model'](input_dict)
        induced_probs = torch.softmax(output_dict['logits'], dim=1)[0]
        exp_entropy = 0.
        for c in range(num_classes):
            labels = torch.tensor([c]).long().to(device)
            retrain_model.train()
            for _ in range(max(int(train_config['num_epochs'] / 10), 5)):
                for input_dict in new_train_dataloader:
                    input_dict = util.to_device(input_dict, device)

                    optimizer.zero_grad()
                    output_dict = retrain_model(input_dict)
                    output_dict['labels'] = input_dict['labels']

                    indices = input_dict['indices']
                    idx = torch.where(indices == unlabeled_idx)[0]
                    if len(idx) > 0:
                        output_dict['labels'][idx] = labels

                    losses = criterion(output_dict)
                    loss = losses['loss']
                    loss.backward()
                    optimizer.step()

            retrain_model.eval()
            with torch.no_grad():
                entropy = 0.
                for j, unlabeled_input_dict in enumerate(unlabeled_dataloader):
                    if i == j: continue
                    unlabeled_input_dict = util.to_device(unlabeled_input_dict, device)

                    output_dict = retrain_model(unlabeled_input_dict)
                    probs = torch.softmax(output_dict['logits'], dim=1)
                    entropy += (-torch.sum(probs * torch.log(probs)))

                exp_entropy += induced_probs[c] * entropy

        if i % 100 == 0:
            logger.info('Retraining is done - {}/{}'.format(i, len(subset)))

        total_exp_entropy.append(exp_entropy)

    return torch.tensor(total_exp_entropy)


# kmeans ++ initialization
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0:
            import IPython; IPython.embed()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll: ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll


class Strategy:
    def __init__(self, net, num_subset, num_classes=10):
        self.net = net
        self.num_subset = num_subset
        self.num_classes = num_classes
        use_cuda = torch.cuda.is_available()

    def query(self, n):
        pass

    def get_grad_embedding(self, loader_te):
        self.net.eval()
        embedding = np.zeros([self.num_subset, self.embedding_dim * self.num_classes])
        idx = 0
        with torch.no_grad():
            for input_dict in loader_te:
                batch_size = input_dict['labels'].shape[0]
                input_dict = util.to_device(input_dict, next(self.net.parameters()).device)
                output_dict = self.net(input_dict)
                cout, out = output_dict['logits'], output_dict['embedding']
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs,1)
                for j in range(batch_size):
                    for c in range(self.num_classes):
                        if c == maxInds[j]:
                            embedding[idx][self.embedding_dim * c : self.embedding_dim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idx][self.embedding_dim * c : self.embedding_dim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
                    idx += 1
            return torch.Tensor(embedding)


class BadgeSampling(Strategy):
    def __init__(self, net, data_config, model_config):
        num_subset = data_config['al_params']['num_subset']
        num_classes = model_config['model_arch']['num_classes']
        super(BadgeSampling, self).__init__(net, num_subset, num_classes)
        self.embedding_dim = net.linear.weight.shape[1]

    def query(self, n, loader_te):
        gradEmbedding = self.get_grad_embedding(loader_te).numpy()
        chosen = init_centers(gradEmbedding, n)
        return chosen[::-1]


