import torch
import numpy as np
import os
import random
from PIL import Image

from domainbed.datasets import datasets
from domainbed.lib import misc
from domainbed.datasets import transforms as DBT
from torchvision import transforms as T
from torchvision.transforms.functional import rotate
global_train_set = []
def set_transfroms(dset, data_type, hparams, algorithm_class=None):
    """
    Args: data_type: ['train', 'valid', 'test', 'mnist']
    """
    assert hparams["data_augmentation"]

    additional_data = False
    if data_type == "train":
        dset.transforms = {"x": DBT.aug}
        additional_data = True
    elif data_type == "valid":
        if hparams["val_augment"] is False:
            #doing
            dset.transforms = {"x": DBT.basic}
        else:
            # Originally, DomainBed use same training augmentation policy to validation.
            # We turn off the augmentation for validation as default,
            # but left the option to reproducibility.
            dset.transforms = {"x": DBT.aug}
    elif data_type == "test":
        dset.transforms = {"x": DBT.basic}
    elif data_type == "mnist":
        # No augmentation for mnist
        dset.transforms = {"x": lambda x: x}
    else:
        raise ValueError(data_type)

    if additional_data and algorithm_class is not None:
        for key, transform in algorithm_class.transforms.items():
            dset.transforms[key] = transform


def get_dataset(test_envs, args, hparams, algorithm_class=None):
    """Get dataset and split."""
    is_mnist = "MNIST" in args.dataset
    dataset = vars(datasets)[args.dataset](args.data_dir)
    #  if not isinstance(dataset, MultipleEnvironmentImageFolder):
    #      raise ValueError("SMALL image datasets are not implemented (corrupted), for transform.")
    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset):
        # The split only depends on seed_hash (= trial_seed). It means that the split is always identical only if use same trial_seed, independent to run the code where, when, or how many times.
        out, in_= split_dataset(env,int(len(env) * args.holdout_fraction),test_envs=test_envs[0], seed=misc.seed_hash(args.trial_seed, env_i))#args.holdout_fraction0.2
        #out: eval; in:train
        if env_i in test_envs:
            in_type = "test"
            out_type = "test"
        else:
            in_type = "train"
            out_type = "valid"
        if is_mnist:
            in_type = "mnist"
            out_type = "mnist"
        set_transfroms(in_, in_type, hparams, algorithm_class)
        set_transfroms(out, out_type, hparams, algorithm_class)

        if hparams["class_balanced"]:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
        else:
            in_weights, out_weights = None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
    return dataset, in_splits, out_splits

def get_dataset_1(test_envs, args, hparams, algorithm_class=None):
    """Get dataset and split."""
    is_mnist = "MNIST" in args.dataset
    dataset = vars(datasets)[args.dataset](args.data_dir)
    #  if not isinstance(dataset, MultipleEnvironmentImageFolder):
    #      raise ValueError("SMALL image datasets are not implemented (corrupted), for transform.")
    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset):
        # The split only depends on seed_hash (= trial_seed). It means that the split is always identical only if use same trial_seed, independent to run the code where, when, or how many times.
        out, in_= split_dataset_1(env,int(len(env) * args.holdout_fraction),test_envs=test_envs[0], seed=misc.seed_hash(args.trial_seed, env_i))#args.holdout_fraction0.2
        #out: eval; in:train
        if env_i in test_envs:
            in_type = "test"
            out_type = "test"
        else:
            in_type = "train"
            out_type = "valid"
        if is_mnist:
            in_type = "mnist"
            out_type = "mnist"
        set_transfroms(in_, in_type, hparams, algorithm_class)
        set_transfroms(out, out_type, hparams, algorithm_class)

        if hparams["class_balanced"]:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
        else:
            in_weights, out_weights = None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
    return dataset, in_splits, out_splits


class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys, test_envs):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset#
        self.domain_ori = [0, 15, 30, 45, 60, 75]
        self.d =  [0, 15, 30, 45, 60, 75]
        del self.d[test_envs]
        self.transforms = {}
        self.direct_return = isinstance(underlying_dataset, _SplitDataset)
        self.keys = keys
        self.flag = (len(keys)/len(underlying_dataset)) > 0.5


    def __getitem__(self, key):
        if self.direct_return:
            return self.underlying_dataset[self.keys[key]]

        x, y, d = self.underlying_dataset[self.keys[key]]
        other = random.sample([k for k in self.keys if k != key],1)

        x_1,y_1,d_1 = self.underlying_dataset[other[0]]
        while y_1 != y:
            other = random.sample([k for k in self.keys if k != key],1)
            x_1,y_1,d_1 = self.underlying_dataset[other[0]]
        other = random.sample([k for k in self.keys if k != key],1)
        x_2,y_2,d_2 = self.underlying_dataset[other[0]]
        while y_2 != y:
            other = random.sample([k for k in self.keys if k != key],1)
            x_2,y_2,d_2 = self.underlying_dataset[other[0]]
        ret = {"y": y}
        if self.flag:
            domain_set = self.d
            if d in domain_set:
                del domain_set[domain_set.index(d)]
            result = [x1 - d for x1 in domain_set]
            angle = [value for value in result  if value != 0]
            ret["x_same"] = x_1
            temp_angle = random.choice(angle)
            rotation = T.Compose([T.ToPILImage(),T.Lambda(lambda x: rotate(x,int(temp_angle), fill=(0,), resample=Image.BICUBIC)),T.ToTensor(),])
            ret["x_diff"] =rotation(x_2)
            ret["d_x"]      = self.domain_ori.index(int(d))
            ret["d_same"]   = self.domain_ori.index(int(d))
            ret["d_diff"]   = self.domain_ori.index(int(d_2+temp_angle))
            

        for key, transform in self.transforms.items():
            ret[key] = transform(x)

        return ret

    def __len__(self):
        return len(self.keys)
    
def refresh():
    global_train_set.clear()
    return 0 

def split_dataset(dataset, n, test_envs, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert n <= len(dataset)
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1, test_envs), _SplitDataset(dataset, keys_2, test_envs)

def split_dataset_1(dataset, n, test_envs, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert n <= len(dataset)
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    
    return _SplitDataset(dataset, keys_1, test_envs), _SplitDataset(dataset, keys_2, test_envs)
