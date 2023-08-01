# custom data samplers
import torch


class CyclicRandomSampler(object):
    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        while True:
            epoch = self.epoch + 1
            self.set_epoch(epoch)
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
            for idx in indices:
                yield idx

    def set_epoch(self, epoch):
        self.epoch = epoch

