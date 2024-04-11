# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
import copy
from typing import List
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from timm.models.layers import trunc_normal_
from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches
from domainbed.optimizers import get_optimizer

from domainbed.models.resnet_mixstyle import (
    resnet18_mixstyle_L234_p0d5_a0d1,
    resnet50_mixstyle_L234_p0d5_a0d1,
)
from domainbed.models.resnet_mixstyle2 import (
    resnet18_mixstyle2_L234_p0d5_a0d1,
    resnet50_mixstyle2_L234_p0d5_a0d1,
)

#from domainbed.losses import  ProxyLoss, ProxyPLoss

class BaseSimCLRException(Exception):
    """Base exception"""

class InvalidBackboneError(BaseSimCLRException):
    """Raised when the choice of backbone Convnet is invalid."""


def to_minibatch(x, y):
    minibatches = list(zip(x, y))
    return minibatches

    
class Classifier(nn.Module):
    def __init__(self,input_num,class_num):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_num, class_num)

    def forward(self,x):
        x = self.fc(x)
        return x

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    transforms = {}
    
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.input_shape = input_shape
        
        print("==============input-shape==========", self.input_shape)
        
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.hparams = hparams

    def update(self, x, y, **kwargs):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.predict(x)

    def new_optimizer(self, parameters):
        optimizer = get_optimizer(
            self.hparams["optimizer"],
            parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        return optimizer

    def clone(self):
        clone = copy.deepcopy(self)
        clone.optimizer = self.new_optimizer(clone.network.parameters())
        clone.optimizer.load_state_dict(self.optimizer.state_dict())

        return clone


class DDneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(DDneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out



class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.encoder, self.scale, self.pcl_weights = networks.encoder(hparams)
        self._initialize_weights(self.encoder)
        self.fea_proj, self.fc_proj = networks.fea_proj(hparams)
        nn.init.kaiming_uniform_(self.fc_proj, mode='fan_out', a=math.sqrt(5))
        self.featurizer = networks.Featurizer(input_shape, self.hparams)

        self.classifier0 = nn.Parameter(torch.FloatTensor(num_classes, self.hparams['out_dim']))
        nn.init.kaiming_uniform_(self.classifier0, mode='fan_out', a=math.sqrt(5))
        self.classifier1 = nn.Parameter(torch.FloatTensor(num_classes, self.hparams['out_dim']))
        nn.init.kaiming_uniform_(self.classifier1, mode='fan_out', a=math.sqrt(5))
        self.classifier2 = nn.Parameter(torch.FloatTensor(num_classes, self.hparams['out_dim']))
        nn.init.kaiming_uniform_(self.classifier2, mode='fan_out', a=math.sqrt(5))
        self.classifier3 = nn.Parameter(torch.FloatTensor(num_classes, self.hparams['out_dim']))
        nn.init.kaiming_uniform_(self.classifier3, mode='fan_out', a=math.sqrt(5))

        self.optimizer = torch.optim.Adam([
            {'params': self.featurizer.parameters()},
            {'params': self.encoder.parameters()},
        	{'params': self.fea_proj.parameters()},
        	{'params': self.fc_proj},
        	{'params': self.classifier0},
            {'params': self.classifier1},
            {'params': self.classifier2},
            {'params': self.classifier3},
        ], lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"])

    def _initialize_weights(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        rep, pred = self.predict(all_x)
        loss_cls = F.nll_loss(F.log_softmax(pred, dim=1), all_y)
        #fc_proj = F.linear(self.classifier0, self.fc_proj)
        #assert fc_proj.requires_grad == True
        #loss_pcl = self.proxycloss(rep, all_y, fc_proj)
        
        loss = loss_cls 

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss_cls": loss_cls.item()}

    def predict(self, x):
        x = self.encoder(x)
        rep = self.fea_proj(x)
        pred0 = F.linear(x, self.classifier0)
        return rep, pred0



class ARM(ERM):
    """Adaptive Risk Minimization (ARM)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        original_input_shape = input_shape
        input_shape = (1 + original_input_shape[0],) + original_input_shape[1:]
        super(ARM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.context_net = networks.ContextNet(original_input_shape)
        self.support_size = hparams["batch_size"]

    def predict(self, x):
        batch_size, c, h, w = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        context = self.context_net(x)
        context = context.reshape((meta_batch_size, support_size, 1, h, w))
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x = torch.cat([x, context], dim=1)
        return self.network(x)


class SAM(ERM):
    """Sharpness-Aware Minimization
    """
    @staticmethod
    def norm(tensor_list: List[torch.tensor], p=2):
        """Compute p-norm for tensor list"""
        return torch.cat([x.flatten() for x in tensor_list]).norm(p)

    def update(self, x, y, **kwargs):
        all_x = torch.cat([xi for xi in x])
        all_y = torch.cat([yi for yi in y])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        # 1. eps(w) = rho * g(w) / g(w).norm(2)
        #           = (rho / g(w).norm(2)) * g(w)
        grad_w = autograd.grad(loss, self.network.parameters())
        scale = self.hparams["rho"] / self.norm(grad_w)
        eps = [g * scale for g in grad_w]

        # 2. w' = w + eps(w)
        with torch.no_grad():
            for p, v in zip(self.network.parameters(), eps):
                p.add_(v)

        # 3. w = w - lr * g(w')
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        # restore original network params
        with torch.no_grad():
            for p, v in zip(self.network.parameters(), eps):
                p.sub_(v)
        self.optimizer.step()

        return {"loss": loss.item()}


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams, conditional, class_balance):

        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.register_buffer("update_count", torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.discriminator = networks.MLP(self.featurizer.n_outputs, num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes, self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = get_optimizer(
            hparams["optimizer"],
            (list(self.discriminator.parameters()) + list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams["weight_decay_d"],
            betas=(self.hparams["beta1"], 0.9),
        )

        self.gen_opt = get_optimizer(
            hparams["optimizer"],
            (list(self.featurizer.parameters()) + list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams["weight_decay_g"],
            betas=(self.hparams["beta1"], 0.9),
        )

    def update(self, x, y, **kwargs):
        self.update_count += 1
        all_x = torch.cat([xi for xi in x])
        all_y = torch.cat([yi for yi in y])
        minibatches = to_minibatch(x, y)
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat(
            [
                torch.full((x.shape[0],), i, dtype=torch.int64, device="cuda")
                for i, (x, y) in enumerate(minibatches)
            ]
        )

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1.0 / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction="none")
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(
            disc_softmax[:, disc_labels].sum(), [disc_input], create_graph=True
        )[0]
        grad_penalty = (input_grad ** 2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams["grad_penalty"] * grad_penalty

        d_steps_per_g = self.hparams["d_steps_per_g_step"]
        if self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g:

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {"disc_loss": disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = classifier_loss + (self.hparams["lambda"] * -disc_loss)
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {"gen_loss": gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class DANN(AbstractDANN):
    """Unconditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DANN, self).__init__(
            input_shape,
            num_classes,
            num_domains,
            hparams,
            conditional=False,
            class_balance=False,
        )


class CDANN(AbstractDANN):
    """Conditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CDANN, self).__init__(
            input_shape,
            num_classes,
            num_domains,
            hparams,
            conditional=True,
            class_balance=True,
        )


class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("update_count", torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        scale = torch.tensor(1.0).cuda().requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        penalty_weight = (
            self.hparams["irm_lambda"]
            if self.update_count >= self.hparams["irm_penalty_anneal_iters"]
            else 1.0
        )
        nll = 0.0
        penalty = 0.0

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx : all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams["irm_penalty_anneal_iters"]:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = get_optimizer(
                self.hparams["optimizer"],
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {"loss": loss.item(), "nll": nll.item(), "penalty": penalty.item()}


class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(VREx, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("update_count", torch.tensor([0]))

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0

        nll = 0.0

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx : all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams["vrex_penalty_anneal_iters"]:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = get_optimizer(
                self.hparams["optimizer"],
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {"loss": loss.item(), "nll": nll.item(), "penalty": penalty.item()}


class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Mixup, self).__init__(input_shape, num_classes, num_domains, hparams)

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        objective = 0

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"], self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {"loss": objective.item()}


class OrgMixup(ERM):
    """
    Original Mixup independent with domains
    """

    def update(self, x, y, **kwargs):
        x = torch.cat(x)
        y = torch.cat(y)

        indices = torch.randperm(x.size(0))
        x2 = x[indices]
        y2 = y[indices]

        lam = np.random.beta(self.hparams["mixup_alpha"], self.hparams["mixup_alpha"])

        x = lam * x + (1 - lam) * x2
        predictions = self.predict(x)

        objective = lam * F.cross_entropy(predictions, y)
        objective += (1 - lam) * F.cross_entropy(predictions, y2)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {"loss": objective.item()}


class CutMix(ERM):
    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def update(self, x, y, **kwargs):
        # cutmix_prob is set to 1.0 for ImageNet and 0.5 for CIFAR100 in the original paper.
        x = torch.cat(x)
        y = torch.cat(y)

        r = np.random.rand(1)
        if self.hparams["beta"] > 0 and r < self.hparams["cutmix_prob"]:
            # generate mixed sample
            beta = self.hparams["beta"]
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(x.size()[0]).cuda()
            target_a = y
            target_b = y[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
            x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
            # compute output
            output = self.predict(x)
            objective = F.cross_entropy(output, target_a) * lam + F.cross_entropy(
                output, target_b
            ) * (1.0 - lam)
        else:
            output = self.predict(x)
            objective = F.cross_entropy(output, y)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {"loss": objective.item()}


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q) / len(minibatches)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}


class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains, hparams)

    def update(self, x, y, **kwargs):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)

            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)

            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)

        That is, when calling .step(), we want grads to be Gi + beta * Gj

        For computational efficiency, we do not compute second derivatives.
        """
        minibatches = to_minibatch(x, y)
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            # fine tune clone-network on task "i"
            inner_net = copy.deepcopy(self.network)

            inner_opt = get_optimizer(
                self.hparams["optimizer"],
                #  "SGD",
                inner_net.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # 1. Compute supervised loss for meta-train set
            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(), inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()

            # 2. Compute meta loss for meta-val set
            # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(), allow_unused=True)

            # `objective` is populated for reporting purposes
            objective += (self.hparams["mldg_beta"] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(self.hparams["mldg_beta"] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)

        self.optimizer.step()

        return {"loss": objective}

class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains, hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(
            x1_norm
        )
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=(0.001, 0.01, 0.1, 1, 10, 100, 1000)):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= nmb * (nmb - 1) / 2

        self.optimizer.zero_grad()
        (objective + (self.hparams["mmd_gamma"] * penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {"loss": objective.item(), "penalty": penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian=False)


class MTL(Algorithm):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning
    (https://arxiv.org/abs/1711.07910)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MTL, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs * 2, num_classes)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        self.register_buffer("embeddings", torch.zeros(num_domains, self.featurizer.n_outputs))

        self.ema = self.hparams["mtl_ema"]

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x, env), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = self.ema * return_embedding + (1 - self.ema) * self.embeddings[env]

            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))


class SagNet(Algorithm):
    """
    Style Agnostic Network
    Algorithm 1 from: https://arxiv.org/abs/1910.11645
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SagNet, self).__init__(input_shape, num_classes, num_domains, hparams)
        # featurizer network
        self.network_f = networks.Featurizer(input_shape, self.hparams)
        # content network
        self.network_c = nn.Linear(self.network_f.n_outputs, num_classes)
        # style network
        self.network_s = nn.Linear(self.network_f.n_outputs, num_classes)

        # # This commented block of code implements something closer to the
        # # original paper, but is specific to ResNet and puts in disadvantage
        # # the other algorithms.
        # resnet_c = networks.Featurizer(input_shape, self.hparams)
        # resnet_s = networks.Featurizer(input_shape, self.hparams)
        # # featurizer network
        # self.network_f = torch.nn.Sequential(
        #         resnet_c.network.conv1,
        #         resnet_c.network.bn1,
        #         resnet_c.network.relu,
        #         resnet_c.network.maxpool,
        #         resnet_c.network.layer1,
        #         resnet_c.network.layer2,
        #         resnet_c.network.layer3)
        # # content network
        # self.network_c = torch.nn.Sequential(
        #         resnet_c.network.layer4,
        #         resnet_c.network.avgpool,
        #         networks.Flatten(),
        #         resnet_c.network.fc)
        # # style network
        # self.network_s = torch.nn.Sequential(
        #         resnet_s.network.layer4,
        #         resnet_s.network.avgpool,
        #         networks.Flatten(),
        #         resnet_s.network.fc)

        def opt(p):
            return get_optimizer(
                hparams["optimizer"], p, lr=hparams["lr"], weight_decay=hparams["weight_decay"]
            )

        self.optimizer_f = opt(self.network_f.parameters())
        self.optimizer_c = opt(self.network_c.parameters())
        self.optimizer_s = opt(self.network_s.parameters())
        self.weight_adv = hparams["sag_w_adv"]

    def forward_c(self, x):
        # learning content network on randomized style
        return self.network_c(self.randomize(self.network_f(x), "style"))

    def forward_s(self, x):
        # learning style network on randomized content
        return self.network_s(self.randomize(self.network_f(x), "content"))

    def randomize(self, x, what="style", eps=1e-5):
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1).cuda()

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0])
        if what == "style":
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, x, y, **kwargs):
        all_x = torch.cat([xi for xi in x])
        all_y = torch.cat([yi for yi in y])

        # learn content
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # learn style
        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()

        # learn adversary
        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {
            "loss_c": loss_c.item(),
            "loss_s": loss_s.item(),
            "loss_adv": loss_adv.item(),
        }

    def predict(self, x):
        return self.network_c(self.network_f(x))


class RSC(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RSC, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.drop_f = (1 - hparams["rsc_f_drop_factor"]) * 100
        self.drop_b = (1 - hparams["rsc_b_drop_factor"]) * 100
        self.num_classes = num_classes

    def update(self, x, y, **kwargs):
        # inputs
        all_x = torch.cat([xi for xi in x])
        # labels
        all_y = torch.cat([yi for yi in y])
        # one-hot labels
        all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
        # features
        all_f = self.featurizer(all_x)
        # predictions
        all_p = self.classifier(all_f)

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.cuda()).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = self.classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        all_p_muted_again = self.classifier(all_f * mask)

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}
        
class DD_LossFunc(nn.Module):
    def __init__(self,device,temperature=0.07):
        super(DD_LossFunc, self).__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = torch.nn.LogSoftmax(dim=1).to(self.device)

    def init_k(self,x):
        return torch.nn.init.kaiming_uniform_(x)

    def replace_with_eye(self,matrix, start_row, start_col):
        size = matrix.size(0) // 15
        gap = int(6*size)
        eye_matrix = torch.eye(size)
        matrix[start_row:start_row+size, gap+start_col:gap+start_col+size] = eye_matrix
        matrix[gap+start_row:gap+start_row+size, start_col:start_col+size] = eye_matrix
        zero_matrix = torch.zeros(size)
        matrix[start_row:start_row+size, start_col:start_col+size] = zero_matrix
        matrix[gap+start_row:gap+start_row+size, gap+start_col:gap+start_col+size] = zero_matrix
        return matrix

    def forward(self, features,labels,domains,same_d,mark):
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        batch_size =  features.shape[0]//3
        temp_1 =  torch.cat((torch.eye(batch_size), torch.zeros(batch_size,batch_size), torch.zeros(batch_size,batch_size)), dim=0).to(self.device)
        temp_2 =  torch.cat((torch.zeros(batch_size,batch_size), torch.eye(batch_size), torch.zeros(batch_size,batch_size)), dim=0).to(self.device)
        temp_3 =  torch.cat((torch.zeros(batch_size,batch_size), torch.zeros(batch_size,batch_size), torch.eye(batch_size)), dim=0).to(self.device)

        if same_d:
            pos_U = torch.cat((temp_1,temp_2,temp_3), dim=1)
            pos_U = self.replace_with_eye(pos_U, int((mark-1)*batch_size/3), int((mark-1)*batch_size/3))
            w = (~torch.eq(labels,labels.T)|(~torch.eq(domains,domains.T))).float()       
        else:
            pos_U = torch.cat((temp_2,temp_1,temp_2), dim=1)
            w = torch.logical_and(~torch.eq(labels,labels.T),~torch.eye(3*batch_size).bool().to(self.device)).float()        
        pos_mask = (pos_U == 1)
        m2 = w.sum(axis= 0).to(self.device)
        
        for i in range (0,3*batch_size):
            w[i,:] = w[i,:]*(3*batch_size)/m2[i]
        w = torch.log(w)
        similarity_matrix =  similarity_matrix/self.temperature
        positives = similarity_matrix[pos_mask].view(similarity_matrix.shape[0], -1)
        similarity_matrix = similarity_matrix+w
        negatives = similarity_matrix[~pos_mask].view(similarity_matrix.shape[0], -1)
        logits = torch.cat([positives, negatives], dim=1).to(self.device)
        loss = -self.criterion(logits)
        loss = loss.mean(axis = 0)[0]
        return loss
    
class DDIR(Algorithm):
    """
    Domain-Disentangled Invariant Representation Learning [5 source domains]
    """
    def __init__(self, input_shape, nc, num_domains,hps,device,target):
        super(DDIR, self).__init__(input_shape, nc, num_domains, hps)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.CE = nn.CrossEntropyLoss().to(device)
        self.DD = DD_LossFunc(device=device).to(device)
        self.nd = num_domains
        self.device = device
        self.Threshold = 0.75
        self.target = [0,1,2,3,4,5] ##
        del self.target[target]
        print("Source domain:",self.target)
        self.encoder1,_,_=networks.encoder(hps);self._init(self.encoder1);self.fea_proj1, self.fc_proj1=networks.fea_proj(hps);nn.init.kaiming_uniform_(self.fc_proj1, mode='fan_out', a=math.sqrt(5))
        self.encoder2,_,_=networks.encoder(hps);self._init(self.encoder2);self.fea_proj2, self.fc_proj2=networks.fea_proj(hps);nn.init.kaiming_uniform_(self.fc_proj2, mode='fan_out', a=math.sqrt(5))
        self.encoder3,_,_=networks.encoder(hps);self._init(self.encoder3);self.fea_proj3, self.fc_proj3=networks.fea_proj(hps);nn.init.kaiming_uniform_(self.fc_proj3, mode='fan_out', a=math.sqrt(5))
        self.encoder4,_,_=networks.encoder(hps);self._init(self.encoder4);self.fea_proj4, self.fc_proj4=networks.fea_proj(hps);nn.init.kaiming_uniform_(self.fc_proj4, mode='fan_out', a=math.sqrt(5))
        self.encoder5,_,_=networks.encoder(hps);self._init(self.encoder5);self.fea_proj5, self.fc_proj5=networks.fea_proj(hps);nn.init.kaiming_uniform_(self.fc_proj5, mode='fan_out', a=math.sqrt(5)) ##
        self.encoder0,_,_=networks.encoder(hps);self._init(self.encoder0);self.fea_proj0, self.fc_proj0=networks.fea_proj(hps);nn.init.kaiming_uniform_(self.fc_proj0, mode='fan_out', a=math.sqrt(5)) ##
        out = self.hparams['out_dim']
        self.c0 = nn.Parameter(torch.FloatTensor(nc, out));nn.init.kaiming_uniform_(self.c0, mode='fan_out', a=math.sqrt(5))
        self.c1 = nn.Parameter(torch.FloatTensor(nc, out));nn.init.kaiming_uniform_(self.c1, mode='fan_out', a=math.sqrt(5));self.d1=nn.Parameter(torch.FloatTensor(2, out));nn.init.kaiming_uniform_(self.d1, mode='fan_out', a=math.sqrt(5))
        self.c2 = nn.Parameter(torch.FloatTensor(nc, out));nn.init.kaiming_uniform_(self.c2, mode='fan_out', a=math.sqrt(5));self.d2=nn.Parameter(torch.FloatTensor(2, out));nn.init.kaiming_uniform_(self.d2, mode='fan_out', a=math.sqrt(5))
        self.c3 = nn.Parameter(torch.FloatTensor(nc, out));nn.init.kaiming_uniform_(self.c3, mode='fan_out', a=math.sqrt(5));self.d3=nn.Parameter(torch.FloatTensor(2, out));nn.init.kaiming_uniform_(self.d3, mode='fan_out', a=math.sqrt(5))
        self.c4 = nn.Parameter(torch.FloatTensor(nc, out));nn.init.kaiming_uniform_(self.c4, mode='fan_out', a=math.sqrt(5));self.d4=nn.Parameter(torch.FloatTensor(2, out));nn.init.kaiming_uniform_(self.d4, mode='fan_out', a=math.sqrt(5)) ##
        self.c5 = nn.Parameter(torch.FloatTensor(nc, out));nn.init.kaiming_uniform_(self.c5, mode='fan_out', a=math.sqrt(5));self.d5=nn.Parameter(torch.FloatTensor(2, out));nn.init.kaiming_uniform_(self.d5, mode='fan_out', a=math.sqrt(5)) ##
        self.optimizer = torch.optim.Adam([{'params': self.featurizer.parameters()},{'params': self.encoder1.parameters()},{'params': self.fea_proj1.parameters()},{'params': self.fc_proj1},{'params': self.encoder2.parameters()},{'params': self.fea_proj2.parameters()},{'params': self.fc_proj2},{'params': self.encoder3.parameters()},{'params': self.fea_proj3.parameters()},{'params': self.fc_proj3},{'params': self.encoder4.parameters()},{'params': self.fea_proj4.parameters()},{'params': self.fc_proj4},{'params': self.encoder5.parameters()},{'params': self.fea_proj5.parameters()},{'params': self.fc_proj5},{'params': self.encoder0.parameters()},{'params': self.fea_proj0.parameters()},{'params': self.fc_proj0},{'params': self.c0},{'params': self.c1},{'params': self.c2},{'params': self.c3},{'params': self.c4},{'params': self.c5},], lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"])##
        self.optimizer_d = torch.optim.Adam([{'params': self.d1},{'params': self.d2},{'params': self.d3},{'params': self.d4},{'params': self.d5},], lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"]) ##
        
    def _init(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def update(self, x, y,x_diff,x_same,d_x,d_diff,d_same, **kwargs):
        _x = torch.cat(x)
        batch_size = _x.shape[0]
        _diff = torch.cat(x_diff)
        _same = torch.cat(x_same)
        _y = torch.cat(y)
        _d0 = torch.cat(d_x)
        _d1 = torch.cat(d_diff)
        _d2 = torch.cat(d_same)
        all_x = torch.cat((_x ,_diff,_same), dim = 0)
        all_y = torch.cat((_y ,_y, _y), dim = 0).view(1,3*batch_size)
        all_d = torch.cat(( _d0,_d1,_d2), dim = 0).view(1,3*batch_size)
        keys = []
        d_labels = []
        for target_value in self.target:
            keys.append(torch.where(all_d[0] == target_value))
            d_labels.append(torch.where(all_d[0] == target_value, torch.tensor(1).to(self.device), torch.tensor(0).to(self.device)))
        rep,pred,rep1,pred1,rep2,pred2,rep3,pred3,rep4,pred4,rep5,pred5,_,_,_,_,_ = self.predict(all_x)
        
        fc_proj0 = F.linear(self.c0, self.fc_proj0);assert fc_proj0.requires_grad == True
        fc_proj1 = F.linear(self.c1, self.fc_proj1);assert fc_proj1.requires_grad == True
        fc_proj2 = F.linear(self.c2, self.fc_proj2);assert fc_proj2.requires_grad == True
        fc_proj3 = F.linear(self.c3, self.fc_proj3);assert fc_proj3.requires_grad == True
        fc_proj4 = F.linear(self.c4, self.fc_proj4);assert fc_proj4.requires_grad == True ##
        fc_proj5 = F.linear(self.c5, self.fc_proj5);assert fc_proj5.requires_grad == True ##

        loss_0_nce  = self.DD(rep,all_y,all_d, same_d = False, mark = 0);loss_0_cls  = self.CE(pred,all_y[-1])
        loss_d1_nce = self.DD(rep1,all_y,all_d, same_d = True, mark = 1);loss_d1_cls = self.CE(pred1[keys[0]],all_y[-1][keys[0]])
        loss_d2_nce = self.DD(rep2,all_y,all_d, same_d = True, mark = 2);loss_d2_cls = self.CE(pred2[keys[1]],all_y[-1][keys[1]])
        loss_d3_nce = self.DD(rep3,all_y,all_d, same_d = True, mark = 3);loss_d3_cls = self.CE(pred3[keys[2]],all_y[-1][keys[2]])
        loss_d4_nce = self.DD(rep4,all_y,all_d, same_d = True, mark = 4);loss_d4_cls = self.CE(pred4[keys[3]],all_y[-1][keys[3]]) ##          
        loss_d5_nce = self.DD(rep5,all_y,all_d, same_d = True, mark = 5);loss_d5_cls = self.CE(pred5[keys[4]],all_y[-1][keys[4]]) ##
        
        loss = (loss_0_cls + loss_0_nce)* (1- self.nd * self.hparams["alpha"]) + ((loss_d1_nce+loss_d1_cls)+(loss_d2_nce+loss_d2_cls)+(loss_d3_nce+loss_d3_cls)+(loss_d4_nce + loss_d4_cls)+(loss_d5_nce+loss_d5_cls))* self.hparams["alpha"] 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        _,_,_,_,_,_,_,_,_,_,_,_,pred_d1,pred_d2,pred_d3,pred_d4,pred_d5 = self.predict(all_x)    
        loss_d = self.CE(pred_d1,d_labels[0])+self.CE(pred_d2,d_labels[1])+self.CE(pred_d3,d_labels[2])+self.CE(pred_d4,d_labels[3])+self.CE(pred_d5,d_labels[4])
        self.optimizer_d.zero_grad()
        loss_d.backward()
        self.optimizer_d.step()
        return {"loss": loss.item(), "loss_d": loss_d.item()}

    def comp_DS(self,pred1,pred_d1,hp):
        pred1 = F.softmax(pred1, dim=1)
        predd1 = F.softmax(pred_d1, dim=1)
        if hp == "ODS":
            predd1[:, 1] = torch.where(predd1[:, 1] < self.Threshold, torch.tensor(0, dtype=torch.float32).to(self.device), torch.tensor(1, dtype=torch.float32).to(self.device))
            mul_1 = pred1 * predd1[:, 1].view(-1, 1)
        elif hp == "IDS":
            predd1[:, 1] = torch.where(predd1[:, 1] < self.Threshold, torch.tensor(0, dtype=torch.float32).to(self.device), predd1[:, 1])
            mul_1 = pred1 * predd1[:, 1].view(-1, 1)
        return mul_1

    def predict(self, x):
        x_c = self.featurizer(x)
        x0 = self.encoder0(x_c)
        rep0 = self.fea_proj0(x0)
        pred0 = F.linear(x0, self.c0)
        x1 = self.encoder1(x_c)
        rep1 = self.fea_proj1(x1)
        pred1 = F.linear(x1, self.c1)
        pred_d1 = F.linear(x1, self.d1)
        x2 = self.encoder2(x_c)
        rep2 = self.fea_proj2(x2)
        pred2 = F.linear(x2, self.c2)
        pred_d2 = F.linear(x2, self.d2)
        x3 = self.encoder3(x_c)
        rep3 = self.fea_proj3(x3)
        pred3 = F.linear(x3, self.c3)
        pred_d3 = F.linear(x3, self.d3)
        x4 = self.encoder4(x_c)
        rep4 = self.fea_proj4(x4)
        pred4 = F.linear(x4, self.c4)
        pred_d4 = F.linear(x4, self.d4)
        x5 = self.encoder5(x_c)
        rep5 = self.fea_proj5(x5)
        pred5 = F.linear(x5, self.c5)
        pred_d5 = F.linear(x5, self.d5)
        if self.training == True:   ## train
            return rep0,pred0,rep1,pred1,rep2,pred2,rep3,pred3,rep4,pred4,rep5,pred5,pred_d1,pred_d2,pred_d3,pred_d4,pred_d5
        else:                       ## infer
            pred0 = F.softmax(pred0, dim=1)
            mul_1 = self.comp_DS(pred1,pred_d1,self.hparams["DS"]) 
            mul_2 = self.comp_DS(pred2,pred_d2,self.hparams["DS"]) 
            mul_3 = self.comp_DS(pred3,pred_d3,self.hparams["DS"]) 
            mul_4 = self.comp_DS(pred4,pred_d4,self.hparams["DS"]) 
            mul_5 = self.comp_DS(pred5,pred_d5,self.hparams["DS"]) 
            if self.hparams["DS"] == "ODS":
                pred_dd,max_index_d= torch.max(torch.stack([mul_1, mul_2, mul_3, mul_4, mul_5]), dim=0)
                pred  = (1 - self.hparams["alpha1"]) * pred0 + self.hparams["alpha1"] * pred_dd
            elif self.hparams["DS"] == "IDS":
                pred_ec = torch.sum(torch.stack([mul_1, mul_2, mul_3, mul_4, mul_5]), dim=0)
                pred = (1 - self.nd*self.hparams["alpha2"]) * pred0 + self.hparams["alpha2"] * pred_ec
            return rep0,pred