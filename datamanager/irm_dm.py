from copy import deepcopy
from typing import Dict, Tuple
import numpy as np
from torchvision.datasets import MNIST
from fedsim.distributed.data_management import DataManager
from . import utils

def identity_fn(x):
    return x


class IRMDM(DataManager):
    def __init__(
        self,
        root,
        dataset="mnist",
        probability_client1 = 0.2,
        probability_client2 = 0.1,
        probability_test = 0.9,
        label_prob_client1 = 0.25,
        label_prob_client2 = 0.25,
        label_prob_test = 0.25,
        seed=10,
        save_dir=None,
        *args,
        **kwargs,
    ):
        self.dataset_name = dataset
        self.probability_client1 = probability_client1
        self.probability_client2 = probability_client2
        self.probability_test = probability_test
        
        self.label_prob_client1 = label_prob_client1
        self.label_prob_client2 = label_prob_client2
        self.label_test = label_prob_test

        super(IRMDM, self).__init__(root, seed, save_dir, *args, **kwargs)

    
    def make_datasets(
        self, 
        root: str, 
        global_transforms: Dict[str, object]
    ) -> Tuple[object, object]:
        if self.dataset_name == 'colored_mnist':
            oracle_dataset = MNIST(root, train=True, download=True)

            tr_targets = deepcopy(oracle_dataset.targets)
            tr_data = deepcopy(oracle_dataset.data)
            
            # shuffle
            rng_state = np.random.get_state()
            np.random.shuffle(tr_data.numpy())
            np.random.set_state(rng_state)
            np.random.shuffle(tr_targets.numpy())
            
            n = len(tr_targets)

            self.local_data = []
            self.local_targets = []
            
            data_client1, targets_client1 = utils.process(
                tr_data[0:n//2],
                tr_targets[0:n//2],
                self.probability_client1,
                self.label_prob_client1,
            )
            data_client2, targets_client2 = utils.process(
                tr_data[n//2:],
                tr_targets[n//2:],
                self.probability_client2,
                self.label_prob_client2,
            )
            data_test, target_test = utils.process(
                tr_data,
                tr_targets,
                self.probability_client2,
                self.label_prob_client2,
            )
            self.custom_local_data = [data_client1, data_client2]
            self.custom_local_targets = [targets_client1, targets_client2]
            self.custom_oracle_dataset = oracle_dataset
        else:
            raise NotImplementedError

        return dict(train=None), utils.BasicVisionDataset(data_test, target_test)


    def make_transforms(self):
        return identity_fn, identity_fn

    
    def partition_local_data(self, dataset):
        return dict(train=None)


    def get_local_dataset(self, id):
        tr_dset = utils.BasicVisionDataset(
            self.custom_local_data[id],
            self.custom_local_targets[id],
            transform=self.train_transforms,
        )
        return dict(train=tr_dset)
        
    def get_group_dataset(self, ids):
        raise NotImplementedError

    def get_oracle_dataset(self):
        return dict(train=self.custom_oracle_dataset)

    def get_identifiers(self):
        identifiers = [
            self.dataset_name,
            self.probability_client1,
            self.probability_client2,
            self.probability_test,
            self.label_prob_client1,
            self.label_prob_client2,
            self.label_test,
        ]
        for i in range(len(identifiers)):
            identifiers[i] = str(identifiers[i]) 
        
        return identifiers

