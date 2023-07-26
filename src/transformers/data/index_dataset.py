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
import struct
from functools import lru_cache
import numpy as np
import torch


def _warmup_mmap_file(path):
    with open(path, 'rb') as stream:
        while stream.read(100 * 1024 * 1024):
            pass


class Index(object):
    def __init__(self):
        self._HDR_MAGIC = b'MMIDIDX\x00\x00'

        self.sizes = []
        self.split = []
        self.offset = []
        self.num_sent = 0
        self.num_doc = 0
        self.buffer_mmap = None

    def convert_size_to_offset(self, sizes):
        offsets = np.array(sizes, dtype=np.int64) * np.int32().itemsize
        np.cumsum(offsets, axis=0, out=offsets)
        offsets[1:] = offsets[:-1]
        offsets[0] = 0
        return offsets

    def save_index(self, idx_file):
        with open(idx_file, 'wb') as stream:
            stream.write(self._HDR_MAGIC)
            stream.write(struct.pack('<Q', self.num_sent))
            stream.write(struct.pack('<Q', self.num_doc))
            stream.write(np.array(self.sizes, dtype=np.int32).tobytes(order='C'))
            stream.write(np.array(self.split, dtype=np.int32).tobytes(order='C'))

    def build(self, enc_docs, bin_file, idx_file):
        with open(bin_file, 'wb') as stream:
            sum_bytes = 0
            for i, (doc_ids, num_bytes) in enumerate(enc_docs):
                sum_bytes += num_bytes
                if len(doc_ids) == 0:
                    continue

                self.split.append(len(self.sizes))
                self.num_doc += 1

                for sent_ids in doc_ids:
                    np_array = np.array(sent_ids, dtype=np.int32)
                    stream.write(np_array.tobytes(order='C'))
                    self.sizes.append(np_array.size)
                    self.num_sent += 1
        # save dataset index
        self.save_index(idx_file)

        print("total encoded: %.2f Mb text | %d sents | %d docs " % (sum_bytes / 1024 / 1024, self.num_sent, self.num_doc))

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
        self.sizes = np.frombuffer(self.buffer, dtype=np.int32, count=self.num_sent, offset=offset)
        self.split = np.frombuffer(self.buffer, dtype=np.int32, count=self.num_doc,  offset=offset + self.sizes.nbytes)
        self.offset = self.convert_size_to_offset(self.sizes)

    def __del__(self):
        if self.buffer_mmap:
            self.buffer_mmap._mmap.close()
            del self.buffer_mmap

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        return self.offset[i], self.sizes[i]

    def __len__(self):
        return self.num_sent

    @property
    def supports_prefetch(self):
        return False


class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, data_predix, warmup=True):
        super().__init__()
        bin_file = data_predix + '.dat'
        idx_file = data_predix + '.idx'
        self.index = Index()
        self.index.read(idx_file, warmup=warmup)

        if warmup:
            _warmup_mmap_file(bin_file)
        self.buffer_mmap = np.memmap(bin_file, mode='r', order='C')
        self.buffer = memoryview(self.buffer_mmap)

    def __del__(self):
        self.buffer_mmap._mmap.close()
        del self.buffer_mmap
        del self.index

    def __len__(self):
        return len(self.index)

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        offset, size = self.index[idx]
        data = np.frombuffer(self.buffer, dtype=np.int32, count=size, offset=offset)
        return data


class IterableIndexDataset(torch.utils.data.IterableDataset):
    """
        iterable dataset for pretraining with samples of max_seq_len
    """
    def __init__(self, data_predix, max_seq_len, world_size=1, rank=0, warmup=True, seed=1234):
        super().__init__()
        bin_file = data_predix + '.dat'
        idx_file = data_predix + '.idx'
        self.max_seq_len = max_seq_len
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        self.index = Index()
        self.index.read(idx_file, warmup=warmup)

        if warmup:
            _warmup_mmap_file(bin_file)
        self.buffer_mmap = np.memmap(bin_file, mode='r', order='C')
        self.buffer = memoryview(self.buffer_mmap)

    def __del__(self):
        self.buffer_mmap._mmap.close()
        del self.buffer_mmap
        del self.index

    def __iter__(self):
        """
            cyclic itereration
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        while(True):
            # torch generator based on seed and epoch
            self.epoch += 1
            gnr = torch.Generator()
            gnr.manual_seed(self.seed + self.epoch)
            assert len(self.index) >= self.world_size * num_workers, "dataset size less than world_size x num_workers"
            indices = torch.randperm(len(self.index), generator=gnr).tolist()
            indices = indices[((self.rank * num_workers) + worker_id)::(self.world_size * num_workers)]

            sample = []
            length = 0
            segment = []
            segment_id = 0
            for idx in indices:
                offset, size = self.index[idx]
                data = np.frombuffer(self.buffer, dtype=np.int32, count=size, offset=offset)

                ptr = 0
                while ptr < size:
                    need = self.max_seq_len - length
                    rest = size - ptr
                    if need <= rest:
                        sample.append(data[ptr:ptr+need])
                        segment_id += 1
                        segment.append(np.ones((need), dtype=np.int32) * segment_id)
                        ptr = ptr + need
                        input_ids = np.concatenate(sample)
                        attention_mask = np.concatenate(segment)
                        yield {'input_ids': input_ids,
                               'attention_mask': attention_mask,
                               'labels': input_ids.copy().astype(np.int64)}

                        sample = []
                        length = 0
                        segment = []
                        segment_id = 0
                    else:
                        sample.append(data[ptr:])
                        length = length + rest
                        segment_id += 1
                        segment.append(np.ones((rest), dtype=np.int32) * segment_id)
                        ptr = ptr + rest


class MixedIterableIndexDataset(torch.utils.data.IterableDataset):
    """
        mixing multiple iterable datasets according to their weights
    """
    def __init__(self, data_files, weights, max_samples):
        assert len(data_files) == len(weights)
        #
        self.datasets = []
        for data_file in data_files:
            dataset = IterableIndexDataset(data_file, seq_len, warmup=True, seed=123)
            self.datasets.append[iter(dataset)]

        weights = np.array(weights, dtype=np.float64)
        weights /= np.sum(weights)
        self.dataset_index = torch.multinomial(weights, max_samples, replacement=True)

    def __len__(self):
        return len(self.dataset_index)

    def __iter__(self):
        for i in self.dataset_index:
            iter = self.datasets[i]
            data = next(iter)
            yield data
