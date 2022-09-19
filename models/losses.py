import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import diffdist
import torch.distributed as dist
import pdb
import random
from torch.autograd import Variable
from qpth.qp import QPFunction

from scipy.sparse import csr_matrix
import math

def gather(z):
    gather_z = [torch.zeros_like(z) for _ in range(torch.distributed.get_world_size())]
    gather_z = diffdist.functional.all_gather(gather_z, z)
    gather_z = torch.cat(gather_z)

    return gather_z


def accuracy(logits, labels, k):
    topk = torch.sort(logits.topk(k, dim=1)[1], 1)[0]
    labels = torch.sort(labels, 1)[0]
    acc = (topk == labels).all(1).float()
    return acc


def mean_cumulative_gain(logits, labels, k):
    topk = torch.sort(logits.topk(k, dim=1)[1], 1)[0]
    labels = torch.sort(labels, 1)[0]
    mcg = (topk == labels).float().mean(1)
    return mcg


def mean_average_precision(logits, labels, k):
    # TODO: not the fastest solution but looks fine
    argsort = torch.argsort(logits, dim=1, descending=True)
    labels_to_sorted_idx = torch.sort(torch.gather(torch.argsort(argsort, dim=1), 1, labels), dim=1)[0] + 1
    precision = (1 + torch.arange(k, device=logits.device).float()) / labels_to_sorted_idx
    return precision.sum(1) / k

def get_indices_sparse(data):
    cols = np.arange(data.size)
    M = csr_matrix((cols, (data.ravel(), cols)), shape=(int(data.max()) + 1, data.size))
    return [np.unravel_index(row.data, data.shape) for row in M]

class NTXent(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """
    LARGE_NUMBER = 1e9

    def __init__(self, tau=1., gpu=None, multiplier=2, distributed=False):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed
        self.norm = 1.

    def forward(self, z, get_map=False):
        n = z.shape[0]
        assert n % self.multiplier == 0

        z = F.normalize(z, p=2, dim=1) #/ np.sqrt(self.tau)

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # TODO: try to rewrite it with pytorch official tools
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug0>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * (self.multiplier) + m])

            z = torch.cat(z_sorted, dim=0)
            n = z.shape[0]

        n_half = int(n / self.multiplier)
        n_way = int(n / self.multiplier)

        support = z[:n_half].reshape(1, n_way, -1)
        query = z[n_half:].reshape(1, n_way, -1)
        z_batch = torch.cat((support, query), dim = 1)

        logits = torch.bmm(z_batch, torch.transpose(z_batch, 1, 2)) * self.tau
        logits[:, np.arange(n), np.arange(n)] = -self.LARGE_NUMBER

        logprob = F.log_softmax(logits, dim=-1)

        # choose all positive objects for an example, for i it would be (i + k * n/m), where k=0...(m-1)
        m = self.multiplier
        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n//m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)
        # repeat for the batch size
        #labels = np.expand_dims(labels, axis=0)
        #labels = np.repeat(labels, self.bs, axis=0)

        # TODO: maybe different terms for each process should only be computed here...
        loss = -logprob[0, np.repeat(np.arange(n), m-1), labels].sum() / n / (m-1) / self.norm

        # zero the probability of identical pairs
        pred = logprob.data.clone()
        pred[:, np.arange(n), np.arange(n)] = -self.LARGE_NUMBER
        acc = accuracy(pred.reshape(n_way*2, -1), torch.LongTensor(labels.reshape(n, m-1)).to(logprob.device), m-1)

        if get_map:
            _map = mean_average_precision(pred, torch.LongTensor(labels.reshape(n, m-1)).to(logprob.device), m-1)
            return loss, acc, _map

        return loss, acc


class Xent_rot(nn.Module):
    def __init__(self, gpu=None, tau=1, multiplier=2, distributed=False):
        super().__init__()
        #self.head = torch.nn.Linear(128, 4).cuda()
        #self.head.train()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed

    def forward(self, z, get_map=False):
        n = z.shape[0]
        assert n % self.multiplier == 0

        #z = F.normalize(z, p=2, dim=1) / np.sqrt(self.tau)
        #p = F.normalize(p, p=2, dim=1)

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # TODO: try to rewrite it with pytorch official tools
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0, rot0>, <proc0_aug1, rot0>, <proc0_aug0, rot1>, <proc0_aug1, rot1>, ..., <proc0_aug(m-1)>, <proc1_aug0>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(4)]
            # sort it to [<proc0_aug0_r0>, <proc1_aug0_r0>,..,<proc0_aug1_r0>,..,<proc0_aug0_r1>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            for m in range(4):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * (4) + m])

            z = torch.cat(z_sorted, dim=0)
            n = z.shape[0] // 4

        rot_labels = [0, 1, 2, 3]
        rot_labels = torch.tensor(rot_labels * n).cuda()
        rot_labels = rot_labels.reshape(n, 4).T.reshape(-1)

        rand_rot = torch.randperm(n * 4)
        z = z[rand_rot, ...]
        rot_labels = rot_labels[rand_rot]

        rot_logits = z#self.head(z)

        xent_rot = torch.nn.CrossEntropyLoss()
        rot_loss = xent_rot(rot_logits, rot_labels)

        return rot_loss

class Xent_rot_random(nn.Module):
    def __init__(self, gpu=None, tau=1, multiplier=2, distributed=False):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed

    def forward(self, z, labels, get_map=False):
        n = z.shape[0]
        assert n % self.multiplier == 0

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # TODO: try to rewrite it with pytorch official tools
            z_list = diffdist.functional.all_gather(z_list, z)
            z = torch.cat(z_list, dim=0)
            n = z.shape[0]

            l_list = [torch.zeros_like(labels) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # TODO: try to rewrite it with pytorch official tools
            l_list = diffdist.functional.all_gather(l_list, labels)
            l = torch.cat(l_list, dim=0)
        else:
            l = labels

        rot_labels = l

        rand_rot = torch.randperm(n)
        z = z[rand_rot, ...]
        rot_labels = rot_labels[rand_rot]

        rot_logits = z

        xent_rot = torch.nn.CrossEntropyLoss()
        rot_loss = xent_rot(rot_logits, rot_labels)

        return rot_loss


class NTXent_R2D2(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """
    LARGE_NUMBER = 1e9

    def __init__(self, n_supp=1, n_query=1, tau=1., reg=50., gpu=None, multiplier=2, distributed=False):
        super().__init__()
        self.reg = reg
        self.multiplier = multiplier
        self.distributed = distributed
        self.norm = 1.
        self.n_supp = n_supp
        self.n_query = n_query
        self.tau = tau

    def forward(self, z, get_map=False):
        n = z.shape[0]
        assert n % self.multiplier == 0

        z = F.normalize(z, p=2, dim=1) / np.sqrt(self.tau)

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # TODO: try to rewrite it with pytorch official tools
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug0>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * (self.multiplier) + m])

            z = torch.cat(z_sorted, dim=0)
            n = z.shape[0]

        n_way = int(n / self.multiplier)

        support = z[:int(self.n_supp*n_way)].unsqueeze(0)
        query = z[int(self.n_supp*n_way):].unsqueeze(0)

        support_labels = torch.LongTensor(torch.arange(n_way)).unsqueeze(0).cuda()
        support_labels = support_labels.repeat(self.n_supp, 1).unsqueeze(0)
        support_labels = support_labels.reshape(1, -1)

        rand_supp = torch.randperm(n_way * self.n_supp)
        support = support[:, rand_supp, ...]
        support_labels = support_labels[:, rand_supp]

        query_labels = torch.LongTensor(torch.arange(n_way)).unsqueeze(0).cuda()
        query_labels = query_labels.repeat(self.n_query, 1).unsqueeze(0)
        query_labels = query_labels.reshape(1, -1)

        rand_query = torch.randperm(n_way * self.n_query)
        query = query[:, rand_query, ...]
        query_labels = query_labels[:, rand_query]

        logits = R2D2_Woodbury(query, support, support_labels, n_way, self.n_supp, l2_regularizer_lambda=self.reg).squeeze()
        #logits = self.tau * logits
        xent = torch.nn.CrossEntropyLoss()

        loss = xent(logits.squeeze(), query_labels.squeeze().to(logits.device))

        pred = torch.argmax(logits.squeeze(), dim=1).view(-1)
        label = query_labels.squeeze().reshape(-1).to(logits.device)
        acc = pred.eq(label).float().mean()

        if get_map:
            _map = mean_average_precision(pred, torch.LongTensor(labels.reshape(n, m-1)).to(logprob.device), m-1)
            return loss, acc, _map

        return loss, acc


class NTXent_PN(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """
    LARGE_NUMBER = 1e9

    def __init__(self, n_supp=1, n_query=1, tau=1., gpu=None, multiplier=2, distributed=False):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed
        self.norm = 1.
        self.n_supp = n_supp
        self.n_query = n_query

    def forward(self, z, get_map=False):
        n = z.shape[0]
        assert n % self.multiplier == 0

        z = F.normalize(z, p=2, dim=1) / np.sqrt(self.tau)

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # TODO: try to rewrite it with pytorch official tools
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug0>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * (self.multiplier) + m])

            z = torch.cat(z_sorted, dim=0)
            n = z.shape[0]

        n_way = int(n / self.multiplier)

        support = z[:int(self.n_supp*n_way)].unsqueeze(0)
        query = z[int(self.n_supp*n_way):].unsqueeze(0)

        support = support.reshape(1, n_way, -1)
        query = query.reshape(1, n_way, -1)

        support_labels = torch.LongTensor(torch.arange(n_way)).unsqueeze(0).cuda()
        support_labels = support_labels.repeat(1, 1)

        rand = torch.rand(1, n_way)
        rand_supp = rand.argsort(dim=1)
        support[0] = support[0, rand_supp[0], ...]
        support_labels[0] = support_labels[0, rand_supp[0]]

        query_labels = torch.LongTensor(torch.arange(n_way)).unsqueeze(0).cuda()
        query_labels = query_labels.repeat(1, 1)

        rand = torch.rand(1, n_way)
        rand_query = rand.argsort(dim=1)
        query[0] = query[0, rand_query[0], ...]
        query_labels[0] = query_labels[0, rand_query[0]]

        logits = ProtoNetHead(query, support, support_labels, n_way, self.n_supp, normalize=False).squeeze()
        xent = torch.nn.CrossEntropyLoss()
        loss = xent(logits.reshape(-1, n_way), query_labels.reshape(-1).to(logits.device))

        pred = torch.argmax(logits.reshape(-1, n_way), dim=1).view(-1)
        label = query_labels.squeeze().reshape(-1).to(logits.device)
        acc = pred.eq(label).float().mean()

        if get_map:
            _map = mean_average_precision(pred, torch.LongTensor(labels.reshape(n, m-1)).to(logprob.device), m-1)
            return loss, acc, _map

        return loss, acc


class NTXent_R2D2_qcm(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """
    LARGE_NUMBER = 1e9

    def __init__(self, n_supp=1, n_query=1, tau=1., reg=50., gpu=None, multiplier=2, distributed=False):
        super().__init__()
        self.tau = tau
        self.reg = reg
        self.multiplier = multiplier
        self.distributed = distributed
        self.norm = 1.
        self.n_supp = n_supp
        self.n_query = n_query


    def forward(self, z, rand_index, lam, get_map=False):
        n = z.shape[0]
        assert (self.n_supp + self.n_query) == self.multiplier
        assert n % self.multiplier == 0

        z = F.normalize(z, p=2, dim=1)

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # TODO: try to rewrite it with pytorch official tools
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug0>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.multiplier + m])
            z = torch.cat(z_sorted, dim=0)

            idx_list = [torch.zeros_like(rand_index) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # TODO: try to rewrite it with pytorch official tools
            idx_list = diffdist.functional.all_gather(idx_list, rand_index)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug0>, ...]
            #idx_list = [chunk for x in idx_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            #idx_sorted = []
            #for m in range(self.multiplier-1):
            #    for i in range(dist.get_world_size()):
            #        idx_sorted.append(idx_list[i * (self.multiplier - 1) + m])
            #rand_index = torch.cat(idx_sorted, dim=0)
            rand_index = torch.stack(idx_list)

            lam_list = [torch.zeros_like(lam) for _ in range(dist.get_world_size())]
            # TODO: try to rewrite it with pytorch official tools
            lam_list = diffdist.functional.all_gather(lam_list, lam)
            lams = torch.stack(lam_list)


            n = z.shape[0]

        #print('gathered z', z)
        #print('gathered rand_index', rand_index)
        #print(rand_index.shape)
        n_way = int(n / self.multiplier)

        if self.distributed:
            im_per_gpu = int(n_way/dist.get_world_size())

            # gather lams
            gathered_lams = []
            for i in range(dist.get_world_size()):
                gathered_lams += [lams[i]] * im_per_gpu
            gathered_lams = torch.tensor(gathered_lams)
            gathered_lams = gathered_lams.repeat(self.n_query).cuda()

        support = z[:int(self.n_supp*n_way)].unsqueeze(0)
        query = z[int(self.n_supp*n_way):].unsqueeze(0)

        support_labels = torch.LongTensor(torch.arange(n_way)).unsqueeze(0).cuda()
        support_labels = support_labels.repeat(self.n_supp, 1).unsqueeze(0)
        support_labels = support_labels.reshape(1, -1)

        #rand_supp = torch.randperm(n_way * self.n_supp)
        #support = support[:, rand_supp, ...]
        #support_labels = support_labels[:, rand_supp]

        query_labels = torch.LongTensor(torch.arange(n_way)).unsqueeze(0).cuda()
        query_labels_ori = query_labels.data
        query_labels = query_labels.repeat(self.n_query, 1).unsqueeze(0)
        query_labels_a = query_labels.reshape(1, -1)
        
        # gather distributed query_b
        if self.distributed:
            query_labels_b = query_labels_ori.reshape(dist.get_world_size(), -1).repeat(1, self.n_query)
            #print(query_labels_b)
            #print("rand_index", rand_index)
            query_labels_b = query_labels_b[torch.arange(query_labels_b.shape[0]).unsqueeze(-1), rand_index]

            query_labels_b = torch.cat([query_labels_b[:, i*im_per_gpu:(i+1)*im_per_gpu] for i in range(self.n_query)])
            query_labels_b = query_labels_b.reshape(1, -1) 
            #print("b", query_labels_b)
        else:
            query_labels_b = query_labels_a[:, rand_index]

        #rand_query = torch.randperm(n_way * self.n_query)
        #query = query[:, rand_query, ...]

        #query_labels_a = query_labels_a[:, rand_query]
        #query_labels_b = query_labels_b[:, rand_query]

        logits = R2D2_Woodbury(query, support, support_labels, n_way, self.n_supp, l2_regularizer_lambda=self.reg).squeeze()
        logits = self.tau * logits #+ self.bias.cuda()
        #logits = self.scale.cuda() * logits + self.bias.cuda()

        if self.distributed:
            xent = torch.nn.CrossEntropyLoss(reduction='none')
            loss = gathered_lams * xent(logits, query_labels_a.squeeze().to(logits.device)) + (1-gathered_lams) * xent(logits, query_labels_b.squeeze().to(logits.device))
            loss = loss.mean()
            #loss = xent(logits, query_labels_a.squeeze().to(logits.device)) 

            pred = torch.argmax(logits, dim=1).view(-1)
            label_a = query_labels_a.squeeze().to(logits.device)
            label_b = query_labels_b.squeeze().to(logits.device)
            acc = pred.eq(label_a).float() * gathered_lams + (1-gathered_lams) * pred.eq(label_b).float()
            acc = acc.mean()#acc/len(label_a)
        else:
            xent = torch.nn.CrossEntropyLoss()
            loss = lam * xent(logits, query_labels_a.squeeze().to(logits.device)) + (1-lam) * xent(logits, query_labels_b.squeeze().to(logits.device))

            pred = torch.argmax(logits, dim=1).view(-1)
            label_a = query_labels_a.squeeze().to(logits.device)
            label_b = query_labels_b.squeeze().to(logits.device)
            acc = pred.eq(label_a).float().sum() * lam + (1-lam) * pred.eq(label_b).float().sum()
            acc = acc/len(label_a)

        if get_map:
            _map = mean_average_precision(pred, torch.LongTensor(labels.reshape(n, m-1)).to(logprob.device), m-1)
            return loss, acc, _map

        return loss, acc


class NTXent_SVM(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """
    LARGE_NUMBER = 1e9

    def __init__(self, bs=1, n_supp=1, n_query=1, tau=1., reg=50., gpu=None, multiplier=2, distributed=False):
        super().__init__()
        self.tau = tau
        self.reg = reg
        self.multiplier = multiplier
        self.distributed = distributed
        self.norm = 1.
        self.n_supp = n_supp
        self.n_query = n_query
        self.bs = bs

    def forward(self, z, get_map=False):
        n = z.shape[0]
        assert (self.n_supp + self.n_query) == self.multiplier
        assert n % self.multiplier == 0

        z = F.normalize(z, p=2, dim=1)

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # TODO: try to rewrite it with pytorch official tools
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug0>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.multiplier + m])
            z = torch.cat(z_sorted, dim=0)
            n = z.shape[0]

        n_way = int(n / self.multiplier)

        support = z[:int(self.n_supp*n_way)].unsqueeze(0)
        query = z[int(self.n_supp*n_way):].unsqueeze(0)

        batch_size = self.bs
        n_way = int(n_way / batch_size)

        support = support.reshape(batch_size, n_way, -1)
        query = query.reshape(batch_size, n_way, -1)

        support_labels = torch.LongTensor(torch.arange(n_way)).unsqueeze(0).cuda()
        support_labels = support_labels.repeat(batch_size, 1)

        rand = torch.rand(batch_size, n_way)
        rand_supp = rand.argsort(dim=1)
        for ib in range(batch_size):
            support[ib] = support[ib, rand_supp[ib], ...]
            support_labels[ib] = support_labels[ib, rand_supp[ib]]

        query_labels = torch.LongTensor(torch.arange(n_way)).unsqueeze(0).cuda()
        query_labels = query_labels.repeat(batch_size, 1)

        rand = torch.rand(batch_size, n_way)
        rand_query = rand.argsort(dim=1)
        for ib in range(batch_size):
            query[ib] = query[ib, rand_query[ib], ...]
            query_labels[ib] = query_labels[ib, rand_query[ib]]

        logits = MetaOptNetHead_SVM(query, support, support_labels, n_way, self.n_supp).squeeze()
        logits = self.tau * logits #+ self.bias.cuda()
        #logits = self.scale.cuda() * logits + self.bias.cuda()

        xent = torch.nn.CrossEntropyLoss()
        #loss = xent(logits, torch.LongTensor(np.arange(n_half)).to(logits.device))
        loss = xent(logits, query_labels.squeeze().to(logits.device))

        pred = torch.argmax(logits.reshape(-1, n_way), dim=1).view(-1)
        label = query_labels.squeeze().reshape(-1).to(logits.device)
        acc = pred.eq(label).float().mean()

        if get_map:
            _map = mean_average_precision(pred, torch.LongTensor(labels.reshape(n, m-1)).to(logprob.device), m-1)
            return loss, acc, _map

        return loss, acc



def computeGramMatrix(A, B):
    """
    Constructs a linear kernel matrix between A and B.
    We assume that each row in A and B represents a d-dimensional feature vector.
    
    Parameters:
      A:  a (n_batch, n, d) Tensor.
      B:  a (n_batch, m, d) Tensor.
    Returns: a (n_batch, n, m) Tensor.
    """

    assert(A.dim() == 3)
    assert(B.dim() == 3)
    assert(A.size(0) == B.size(0) and A.size(2) == B.size(2))

    return torch.bmm(A, B.transpose(1,2))


def binv(b_mat):
    """
    Computes an inverse of each matrix in the batch.
    Pytorch 0.4.1 does not support batched matrix inverse.
    Hence, we are solving AX=I.
    
    Parameters:
      b_mat:  a (n_batch, n, n) Tensor.
    Returns: a (n_batch, n, n) Tensor.
    """

    id_matrix = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat).cuda()
    b_inv, _ = torch.solve(id_matrix, b_mat)

    return b_inv


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)

    return encoded_indicies

def batched_kronecker(matrix1, matrix2):
    matrix1_flatten = matrix1.reshape(matrix1.size()[0], -1)
    matrix2_flatten = matrix2.reshape(matrix2.size()[0], -1)
    return torch.bmm(matrix1_flatten.unsqueeze(2), matrix2_flatten.unsqueeze(1)).reshape([matrix1.size()[0]] + list(matrix1.size()[1:]) + list(matrix2.size()[1:])).permute([0, 1, 3, 2, 4]).reshape(matrix1.size(0), matrix1.size(1) * matrix2.size(1), matrix1.size(2) * matrix2.size(2))


def R2D2_Woodbury(query, support, support_labels, n_way, n_shot, l2_regularizer_lambda=50.0):
    """
    Fits the support set with ridge regression and
    returns the classification score on the query set.
    This model is the classification head described in:
    Meta-learning with differentiable closed-form solvers
    (Bertinetto et al., in submission to NIPS 2018).
    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      l2_regularizer_lambda: a scalar. Represents the strength of L2 regularization.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """

    device = query.device

    tasks_per_batch = query.size(0)
    n_support = support.size(1)

    support_labels_one_hot = one_hot(support_labels.reshape(tasks_per_batch * n_support), n_way)
    support_labels_one_hot = support_labels_one_hot.reshape(tasks_per_batch, n_support, n_way).cuda()

    id_matrix = torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).cuda()

    # Compute the dual form solution of the ridge regression.
    # W = X^T(X X^T - lambda * I)^(-1) Y
    ridge_sol = computeGramMatrix(support, support) + l2_regularizer_lambda * id_matrix
    ridge_sol = binv(ridge_sol)
    ridge_sol = torch.bmm(support.transpose(1,2), ridge_sol)
    ridge_sol = torch.bmm(ridge_sol, support_labels_one_hot)

    # Compute the classification score.
    # score = W X
    logits = torch.bmm(query, ridge_sol)

    return logits


def R2D2_regression(query, support, support_labels, l2_regularizer_lambda=50.0):
    """
    Fits the support set with ridge regression and
    returns the classification score on the query set.
    This model is the classification head described in:
    Meta-learning with differentiable closed-form solvers
    (Bertinetto et al., in submission to NIPS 2018).
    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      l2_regularizer_lambda: a scalar. Represents the strength of L2 regularization.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """

    device = query.device

    tasks_per_batch = query.size(0)
    n_support = support.size(1)

    id_matrix = torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).cuda()

    # Compute the dual form solution of the ridge regression.
    # W = X^T(X X^T - lambda * I)^(-1) Y
    ridge_sol = computeGramMatrix(support, support) + l2_regularizer_lambda * id_matrix
    ridge_sol = binv(ridge_sol)
    ridge_sol = torch.bmm(support.transpose(1,2), ridge_sol)
    ridge_sol = torch.bmm(ridge_sol, support_labels)

    # Compute the classification score.
    # score = W X
    logits = torch.bmm(query, ridge_sol)

    return logits


def ProtoNetHead(query, support, support_labels, n_way, n_shot, normalize=True):
    """
    Constructs the prototype representation of each class(=mean of support vectors of each class) and 
    returns the classification score (=L2 distance to each class prototype) on the query set.
    
    This model is the classification head described in:
    Prototypical Networks for Few-shot Learning
    (Snell et al., NIPS 2017).
    
    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)
    d = query.size(2)
    
    assert(query.dim() == 3)
    assert(support.dim() == 3)
    assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot
    
    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)
    
    # From:
    # https://github.com/gidariss/FewShotWithoutForgetting/blob/master/architectures/PrototypicalNetworksHead.py
    #************************* Compute Prototypes **************************
    labels_train_transposed = support_labels_one_hot.transpose(1,2)
    # Batch matrix multiplication:
    #   prototypes = labels_train_transposed * features_train ==>
    #   [batch_size x nKnovel x num_channels] =
    #       [batch_size x nKnovel x num_train_examples] * [batch_size * num_train_examples * num_channels]
    prototypes = torch.bmm(labels_train_transposed, support)
    # Divide with the number of examples per novel category.
    prototypes = prototypes.div(
        labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototypes)
    )

    # Distance Matrix Vectorization Trick
    AB = computeGramMatrix(query, prototypes)
    AA = (query * query).sum(dim=2, keepdim=True)
    BB = (prototypes * prototypes).sum(dim=2, keepdim=True).reshape(tasks_per_batch, 1, n_way)
    logits = AA.expand_as(AB) - 2 * AB + BB.expand_as(AB)
    logits = -logits
    
    if normalize:
        logits = logits / d

    return logits

def MetaOptNetHead_SVM(query, support, support_labels, n_way, n_shot, C_reg=0.1, double_precision=False, maxIter=15):
    """
    Fits the support set with multi-class SVM and 
    returns the classification score on the query set.
    
    This is the multi-class SVM presented in:
    On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines
    (Crammer and Singer, Journal of Machine Learning Research 2001).
    This model is the classification head that we use for the final version.
    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      C_reg: a scalar. Represents the cost parameter C in SVM.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)

    assert(query.dim() == 3)
    assert(support.dim() == 3)
    assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
    #assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot

    #Here we solve the dual problem:
    #Note that the classes are indexed by m & samples are indexed by i.
    #min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i
    #s.t.  \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i

    #where w_m(\alpha) = \sum_i \alpha^m_i x_i,
    #and C^m_i = C if m  = y_i,
    #C^m_i = 0 if m != y_i.
    #This borrows the notation of liblinear.
    
    #\alpha is an (n_support, n_way) matrix
    kernel_matrix = computeGramMatrix(support, support)

    id_matrix_0 = torch.eye(n_way).expand(tasks_per_batch, n_way, n_way).cuda()
    block_kernel_matrix = batched_kronecker(kernel_matrix, id_matrix_0)
    #This seems to help avoid PSD error from the QP solver.
    block_kernel_matrix += 1.0 * torch.eye(n_way*n_support).expand(tasks_per_batch, n_way*n_support, n_way*n_support).cuda()
    
    support_labels_one_hot = one_hot(support_labels.reshape(tasks_per_batch * n_support), n_way) # (tasks_per_batch * n_support, n_support)
    support_labels_one_hot = support_labels_one_hot.reshape(tasks_per_batch, n_support, n_way)
    support_labels_one_hot = support_labels_one_hot.reshape(tasks_per_batch, n_support * n_way)
    
    G = block_kernel_matrix
    e = -1.0 * support_labels_one_hot
    #print (G.size())
    #This part is for the inequality constraints:
    #\alpha^m_i <= C^m_i \forall m,i
    #where C^m_i = C if m  = y_i,
    #C^m_i = 0 if m != y_i.
    id_matrix_1 = torch.eye(n_way * n_support).expand(tasks_per_batch, n_way * n_support, n_way * n_support)
    C = Variable(id_matrix_1)
    h = Variable(C_reg * support_labels_one_hot)
    #print (C.size(), h.size())
    #This part is for the equality constraints:
    #\sum_m \alpha^m_i=0 \forall i
    id_matrix_2 = torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).cuda()

    A = Variable(batched_kronecker(id_matrix_2, torch.ones(tasks_per_batch, 1, n_way).cuda()))
    b = Variable(torch.zeros(tasks_per_batch, n_support))
    #print (A.size(), b.size())
    if double_precision:
        G, e, C, h, A, b = [x.double().cuda() for x in [G, e, C, h, A, b]]
    else:
        G, e, C, h, A, b = [x.float().cuda() for x in [G, e, C, h, A, b]]

    # Solve the following QP to fit SVM:
    #        \hat z =   argmin_z 1/2 z^T G z + e^T z
    #                 subject to Cz <= h
    # We use detach() to prevent backpropagation to fixed variables.
    qp_sol = QPFunction(verbose=False, maxIter=maxIter)(G, e.detach(), C.detach(), h.detach(), A.detach(), b.detach())

    # Compute the classification score.
    compatibility = computeGramMatrix(support, query)
    compatibility = compatibility.float()
    compatibility = compatibility.unsqueeze(3).expand(tasks_per_batch, n_support, n_query, n_way)
    qp_sol = qp_sol.reshape(tasks_per_batch, n_support, n_way)
    logits = qp_sol.float().unsqueeze(2).expand(tasks_per_batch, n_support, n_query, n_way)
    logits = logits * compatibility
    logits = torch.sum(logits, 1)

    return logits
