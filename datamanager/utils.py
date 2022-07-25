from copy import deepcopy
import torch

def _xor(a, b):
        return (a - b).abs()

def _bernouli_sample(probability, sample_size):
    return (torch.rand(sample_size) < probability).float()

def process(data, labels, probability, label_prob):
    images = data.reshape((-1, 28, 28))
    labels = (labels < 5).float()

    labels = _xor(labels, _bernouli_sample(label_prob, len(labels)))
    colors = _xor(labels, _bernouli_sample(probability, len(labels)))
    images = torch.stack([images, images], dim=1)
    images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0
    return images.float()/255., labels[:, None] 


class BasicVisionDataset(object):
    def __init__(self, images, targets, transform=None, target_transform=None):
        transform = deepcopy(transform)
        assert len(images) == len(targets)
        self.images = images
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

        self.images = images
        self.targets = targets

    def __getitem__(self, index):
        img, target = self.images[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.targets)