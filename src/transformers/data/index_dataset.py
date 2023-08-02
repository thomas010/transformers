# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# copied from fairseq/fairseq/data/indexed_dataset.py
# Removed IndexedRawTextDataset since it relied on Fairseq dictionary
# other slight modifications to remove fairseq dependencies
# Added document index to index file and made it accessible.
#    An empty sentence no longer separates documents.

# Some of the fixes/improvements are adopted from
# https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/main/megatron/data/indexed_dataset.py

import sys
import os
import math
import time
import struct
import numpy as np
import torch
from functools import lru_cache
from .data_sampler import CyclicRandomSampler


def _warmup_mmap_file(path):
    with open(path, 'rb') as stream:
        while stream.read(100 * 1024 * 1024):
            pass


class Index(object):
    def __init__(self):
        self._HDR_MAGIC = b'MMIDIDX\x00\x00'

        self.sent_sizes = []
        self.doc_sizes = []
        self.num_sent = 0
        self.num_doc = 0
        self.buffer_mmap = None

    def save_index(self, idx_file):
        with open(idx_file, 'wb') as stream:
            stream.write(self._HDR_MAGIC)
            stream.write(struct.pack('<Q', self.num_sent))
            stream.write(struct.pack('<Q', self.num_doc))
            stream.write(np.array(self.sent_sizes, dtype=np.int32).tobytes(order='C'))
            stream.write(np.array(self.doc_sizes, dtype=np.int32).tobytes(order='C'))

    def build(self, enc_docs, bin_file, idx_file):
        with open(bin_file, 'wb') as stream:
            sum_bytes = 0
            sum_sizes = 0
            for i, (doc_ids, num_bytes) in enumerate(enc_docs):
                sum_bytes += num_bytes
                if len(doc_ids) == 0:
                    continue

                for sent_ids in doc_ids:
                    np_array = np.array(sent_ids, dtype=np.int32)
                    stream.write(np_array.tobytes(order='C'))
                    self.sent_sizes.append(np_array.size)
                    self.num_sent += 1
                    sum_sizes += np_array.size

                self.doc_sizes.append(sum_sizes)
                self.num_doc += 1

        print("total encoded: %.2f Mb text | %d sents | %d docs " % (sum_bytes / 1024 / 1024, self.num_sent, self.num_doc))
        self.save_index(idx_file)

    def read(self, idx_file, warmup=True):
        with open(idx_file, 'rb') as stream:
            magic_test = stream.read(9)
            assert self._HDR_MAGIC == magic_test, 'Index file does not match expected format.'
            self.num_sent = struct.unpack('<Q', stream.read(8))[0]
            self.num_doc = struct.unpack('<Q', stream.read(8))[0]
            offset = stream.tell()

        if warmup:
            _warmup_mmap_file(idx_file)
        self.buffer_mmap = np.memmap(idx_file, mode='r', order='C')
        self.buffer = memoryview(self.buffer_mmap)
        self.sent_sizes = np.frombuffer(self.buffer, dtype=np.int32, count=self.num_sent, offset=offset)
        self.doc_sizes = np.frombuffer(self.buffer, dtype=np.int32, count=self.num_doc,  offset=offset + self.sent_sizes.nbytes)

    def __del__(self):
        if self.buffer_mmap is not None:
            self.buffer_mmap._mmap.close()
            del self.buffer_mmap

    def get_sent_sizes(self):
        return self.sent_sizes

    def get_doc_sizes(self):
        return self.doc_sizes

    def get_sent_num(self):
        return self.num_sent

    def get_doc_num(self):
        return self.num_doc


class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, data_predix, warmup=True):
        super().__init__()
        bin_file = data_predix + '.dat'
        idx_file = data_predix + '.idx'
        self.index = Index()
        self.index.read(idx_file, warmup=warmup)
        self.sizes = self.index.get_sent_sizes()
        self.offsets = self.convert_size_to_offset(self.sizes)
        self.num_sent = self.index.get_sent_num()

        if warmup:
            _warmup_mmap_file(bin_file)
        self.buffer_mmap = np.memmap(bin_file, mode='r', order='C')
        self.buffer = memoryview(self.buffer_mmap)

    def convert_size_to_offset(self, sizes):
        offsets = np.array(sizes, dtype=np.int64) * np.int32().itemsize
        np.cumsum(offsets, axis=0, out=offsets)
        offsets[1:] = offsets[:-1]
        offsets[0] = 0
        return offsets

    def __del__(self):
        self.buffer_mmap._mmap.close()
        del self.buffer_mmap
        del self.index

    def __len__(self):
        return self.num_sent

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        offset = self.offsets[idx]
        size = self.sizes[idx]
        data = np.frombuffer(self.buffer, dtype=np.int32, count=size, offset=offset)
        return data


class SliceDataset(torch.utils.data.Dataset):
    def __init__(self, data_predix, slice_size, eod_id, warmup=True):
        super().__init__()
        bin_file = data_predix + '.dat'
        idx_file = data_predix + '.idx'
        self.index = Index()
        self.index.read(idx_file, warmup=warmup)
        self.slice_size = slice_size
        self.sizes = self.index.get_sent_sizes()
        self.offsets = self.convert_size_to_offset(self.sizes, self.slice_size)
        self.num_slice = len(self.offsets)
        self.eod_id = eod_id

        if warmup:
            _warmup_mmap_file(bin_file)
        self.buffer_mmap = np.memmap(bin_file, mode='r', order='C')
        self.buffer = memoryview(self.buffer_mmap)

    def convert_size_to_offset(self, sizes, slice_size):
        sum_sizes = np.sum(np.array(sizes, dtype=np.int64))
        num_slice = int(sum_sizes / slice_size)
        offsets = np.arange(num_slice, dtype=np.int64) * slice_size * np.int32().itemsize
        offsets = offsets.tolist()
        return offsets

    def __del__(self):
        self.buffer_mmap._mmap.close()
        del self.buffer_mmap
        del self.index

    def __len__(self):
        return self.num_slice

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        offset = self.offsets[idx]
        size = self.slice_size
        data = np.frombuffer(self.buffer, dtype=np.int32, count=size, offset=offset)
        mask = data == self.eod_id
        attention_mask = np.cumsum(mask[::-1]) + 1  # 0 will be masked
        return {'input_ids': data,
                'attention_mask': attention_mask,
                'labels': data.copy().astype(np.int64)
                }


class MultiSliceDataset(torch.utils.data.Dataset):
    def __init__(self, data_predixes, weights, num_samples, slice_size, eod_id, seed, warmup=True):
        super().__init__()
        assert len(data_predixes) == len(weights)
        self.datasets = []
        self.samplers = []
        for data_predix in data_predixes:
            dataset = SliceDataset(data_predix, slice_size, eod_id, warmup)
            sampler = CyclicRandomSampler(dataset, seed)
            self.datasets.append(dataset)
            self.samplers.append(iter(sampler))
        weights = torch.tensor(weights, dtype=torch.float64)
        weights /= torch.sum(weights)
        self.weights = weights
        self.num_samples = num_samples
        self.dataset_index, self.dataset_item_index = self.build_dataset_item_index_fast()

    def build_dataset_item_index_fast(self):
        start_time = time.time()
        dataset_index = np.zeros(self.num_samples, dtype=np.uint8)
        dataset_item_index = np.zeros(self.num_samples, dtype=np.int64)

        from transformers.data import data_utils_cpp
        data_utils_cpp.build_dataset_item_indices(dataset_index, self.weights, len(self.datasets), self.num_samples, True)
        for i in range(self.num_samples):
            index = dataset_index[i]
            dataset_item_index[i] = next(self.samplers[index])
        print('time for building multi slice datasets indices: {:.2f} (sec)'.format(time.time() - start_time))
        return dataset_index, dataset_item_index

    def build_dataset_item_index(self):
        dataset_index = []
        dataset_item_index = []
        num_datasets = len(self.datasets)
        numbers = [0] * num_datasets
        weights = self.weights
        for i in range(self.num_samples):
            min_index = 0
            min_differ = numbers[0]/max(i, 1)-weights[0]
            for j in range(1, num_datasets):
                differ = numbers[j]/max(i, 1)-weights[j]
                if differ < min_differ:
                    min_index = j
                    min_differ = differ
            dataset_index.append(min_index)
            dataset_item_index.append(next(self.samplers[min_index]))
        return dataset_index, dataset_item_index

    def __del__(self):
        for dataset in self.datasets:
            del dataset

    def __len__(self):
        return self.num_samples

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        dataset_index = self.dataset_index[idx]
        dataset_item_index = self.dataset_item_index[idx]
        item = self.datasets[dataset_index][dataset_item_index]
        return item
