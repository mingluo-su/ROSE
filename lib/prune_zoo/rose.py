import math
import time

import numpy as np
import torch
import torch.nn as nn
import transformers
from itertools import chain

DEBUG = True

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class ROSE:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0
     

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.H *= self.nsamples / (self.nsamples + tmp)

        inp = inp.type(torch.float32)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()

        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
        self.H += inp.matmul(inp.t())


    def caculate_block_loss(self,score,W,blocksize,sparsity,prune_n,prune_m):
        
        if prune_n != 0:
            blocksize = prune_m
            W_mask = (torch.zeros_like(score) == 1).to(self.dev)
            for ii in range(score.shape[1]):
                if ii % prune_m == 0:
                    tmp = score[:, ii : (ii + prune_m)].float()
                    W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )
            prune_score = score * W_mask  
            reshaped = prune_score.view(self.rows, self.columns // prune_m, prune_m)
            loss_sum = reshaped.sum(dim=2).sum(dim=0)
        else:
            loss_sum = []
            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)

                W1 = W[:, i1:i2].clone()
                tmp = score[:, i1:i2].clone()
        
                thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                mask1 = tmp <= thresh
                prune_score = tmp*mask1

                loss = torch.sum(prune_score)
                loss_sum.append(loss.item())

            loss_sum = torch.tensor(loss_sum, device=self.dev)
        
        return loss_sum
    
    def reorder_group(
       self,loss_sum,blocksize
    ): 
        sorted_indices = torch.argsort(loss_sum, descending=True)  # shape: [32]
        reordered_indices = torch.hstack([torch.arange(x * blocksize, x * blocksize + blocksize) for x in sorted_indices])
        return reordered_indices
    
    def pruning(
        self, sparsity, W,H,prune_n=0, prune_m=0, blocksize=128, percdamp=.01
    ):
    
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        Hinv = torch.linalg.cholesky(H, upper=True)

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d 
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()

        return W


    def fasterprune(
        self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01
    ):
        if prune_m!=0:
            blocksize = prune_m
        W = self.layer.weight.data.clone()

        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        
        W = W.float()
        
        score = (torch.abs(W) * torch.sqrt(self.scaler_row.reshape((1, -1))))
        
        if prune_n != 0:
            blocksize = prune_m
            W_mask = (torch.zeros_like(score) == 1).to(self.dev)
            for ii in range(score.shape[1]):
                if ii % prune_m == 0:
                    tmp = score[:, ii : (ii + prune_m)].float()
                    W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=True)[1],
                            True,
                        )
            prune_score = score * W_mask  
            reshaped = prune_score.view(self.rows, self.columns // prune_m, prune_m)
            loss_sum = reshaped.sum(dim=2).sum(dim=0)
        else: 
            reordered_indices = []
            loss_sum = []
            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)

                W1 = W[:, i1:i2].clone()
                tmp = score[:, i1:i2].clone()
        
                thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                mask1 = tmp <= thresh
                prune_score = tmp*mask1

                column_score = prune_score.sum(0)
                column_order =  torch.argsort(column_score, descending=True)
                reordered_indices.append( (i1 //blocksize) * blocksize + column_order)

                loss = torch.sum(prune_score)
                loss_sum.append(loss.item())

            loss_sum = torch.tensor(loss_sum, device=self.dev)

            reorder_column = torch.cat(reordered_indices)

        block_loss = self.caculate_block_loss(score,W,blocksize,sparsity,prune_n,prune_m)
       
        ste = (torch.max(block_loss) - torch.min(block_loss)) /  torch.max(block_loss)
        if ste  >  0.5:
            reorder_group = self.reorder_group(block_loss,blocksize)
            reordered_indices = reorder_column[reorder_group]  
        else:
            reordered_indices = torch.arange(self.columns,device=self.dev)
            
        H = self.H.clone()
        del self.H
        H = H[:, reordered_indices]
        H = H[reordered_indices, :]
        W = W[:, reordered_indices]

        W = self.pruning(sparsity,W,H,prune_n=prune_n, prune_m=prune_m, blocksize=blocksize, percdamp=percdamp)
        _, inverse_indices = torch.sort(reordered_indices)
        W_restored = W[:, inverse_indices]
   
        torch.cuda.synchronize()
    
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W_restored.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype) 
        
    
    
    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.scaler_row = None
        torch.cuda.empty_cache()
