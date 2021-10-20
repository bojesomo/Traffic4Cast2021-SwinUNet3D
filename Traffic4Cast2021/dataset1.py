import numpy as np
import h5py
from os.path import dirname, basename
import os
import torch
from torch import randperm
from torch._utils import _accumulate
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import time
import copy
from sklearn.model_selection import train_test_split
from .graph_utils import construct_mask
from einops import repeat


class ToCuda(object):
    """ put on cuda """

    def __call__(self, sample):
        if isinstance(sample, tuple):
            X, y = sample
            X = [x.cuda() for x in X]
            y = y.cuda()
            return X, y
        else:  # test case {only X list}
            X = sample
            X = [x.cuda() for x in X]
            return X


class ToTensor(object):
    """ Convert ndarrays in sample to Tensors """

    def __call__(self, sample):
        return tuple(
            [torch.from_numpy(x.transpose(eval(f"{f'0,' if (x.ndim == 4) else f''}-1,-3,-2"))) for x in sample])


class RandomFlip(object):
    def __init__(self, p=0.5, axis=0, collapse_time=True):
        super().__init__()
        self.p = p
        self.axis = axis
        self.collapse_time = collapse_time

    def __call__(self, sample, metadata):
        in_seq, out_seq = sample
        if torch.rand(1) < self.p:
            in_seq = np.flip(in_seq, axis=self.axis).copy()
            out_seq = np.flip(out_seq, axis=self.axis).copy()
        if 'mask' in metadata:
            if self.collapse_time:
                metadata['mask'] = np.flip(metadata['mask'], axis=self.axis).copy()
            else:
                if self.axis > 0:
                    metadata['mask'] = np.flip(metadata['mask'], axis=self.axis - 1).copy()

        return (in_seq, out_seq), metadata

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomHorizontalFlip(RandomFlip):
    def __init__(self, p=0.5, collapse_time=True):
        super().__init__(p=p, axis=1)


class RandomVerticalFlip(RandomFlip):
    def __init__(self, p=0.5, collapse_time=True):
        super().__init__(p=p, axis=0)


class T4CDataset(Dataset):
    """Generates data for PyTorch"""

    def __init__(self, data_root='/l/proj/kuex0005/Alabi/Datasets/Traffic4Cast2021/temporal/BERLIN', n_channels=3,
                 n_channels_out=None, sampling_step=1, n_frame_in=12, times_out=None, collapse_time=True,
                 transform=transforms.Compose([ToTensor()]), use_static=False, use_time_slot=False,
                 stage='training', augment_data=False, **kwargs):
        'Initialization'
        if times_out is None:
            times_out = '5:10:15:30:45:60'
        assert stage in ['training', 'test'], f"stage={stage} is not in ['training', 'test']"

        self.collapse_time = collapse_time
        self.stage = stage
        self.use_time_slot = use_time_slot  # args.use_time_slot
        self.use_static = use_static  # args.use_static
        self.transform = transform
        self.n_frame_in = n_frame_in  # args.n_frame_in
        self.times_out = [int(t) for t in times_out.split(':')]  # in mins
        self.n_frame_out_last = self.times_out[-1] // 5

        self.n_channels = n_channels
        self.n_out_channel = n_channels_out
        self.data_dir = data_root

        if self.stage == 'training':
            self.sampling_step = sampling_step
            self.n_partitions = 288  # daily partitions
            self.parts_per_file = (self.n_partitions - (
                        self.n_frame_in + self.n_frame_out_last)) // self.sampling_step + 1
            # self.file_filter = f"**/{self.stage}/*8ch.h5"
            self.file_filter = f"{self.stage}/*8ch.h5"
            # if use_extra:
            #     self.file_filter.append(f"extra/**/{self.stage}/*8ch.h5")
        else:
            self.sampling_step = 1
            self.parts_per_file = self.n_partitions = 100  # sample test partitions
            self.file_filter = f"*_{self.stage}_*_*.h5"  # using test_additional as filter
            # self.file_filter = [f"{folder}/**/*_{self.stage}_*_*.h5" for folder in
            #                     ['temporal', 'spatiotemporal']]  # using test_additional as filter

        # self.first_term = 0  # for modified data augmentation
        # self.count = 0  # number of samples used already

        # self.files = list(Path(self.data_dir).rglob(self.file_filter))
        self.files = list(Path(self.data_dir).rglob(self.file_filter))
        self.file_index = 0

        self.city = basename(data_root)
        self.city_category = basename(dirname(data_root))

        if self.stage == 'training':
            # self.cities = list(set([basename(str(file)).split('_')[1] for file in self.files]))
            self.data = self.get_data(self.files[0])
        else:
            # self.cities = list(set([basename(str(file)).split('_')[0] for file in self.files]))
            self.data_additional = self.get_data(self.files[0])
            self.data = self.get_data(str(self.files[0]).replace('_additional', ''))

        static_file = os.path.join(data_root, f"{self.city}_static.h5")
        # self.static_data = self.get_data(static_file).transpose(1, 2, 0)  # transposed for ease
        # self.static_mask = construct_mask(self.get_data(static_file))
        static_data = self.get_data(static_file)
        self.static_mask = construct_mask(static_data)
        self.static_map = np.expand_dims(static_data[0, ...], -1)

        self.augmentation = None
        if stage == 'training' and augment_data:
            if collapse_time:
                self.augmentation = [RandomVerticalFlip(), RandomHorizontalFlip()]
            else:
                self.augmentation = [RandomFlip(axis=i, collapse_time=collapse_time) for i in range(3)]

        self.indices = np.arange(len(self.files) * self.parts_per_file)

    def __len__(self):
        """Denotes the number of samples per epoch"""
        return len(self.indices)

    def do_augmentation(self, sample, metadata):
        if self.augmentation is not None:
            for augment_fn in self.augmentation:
                sample, metadata = augment_fn(sample, metadata)
        return sample, metadata

    def __getothers(self, index):
        """Generate one batch of data"""

        # Generate data
        sample, metadata = self.__data_generation(index)
        sample, metadata = self.do_augmentation(sample, metadata)
        if self.transform:
            sample = self.transform(sample)

        return sample, metadata

    def __getitem__(self, index):
        """Generate one batch of data"""
        if index > self.__len__():
            raise IndexError("Index out of bounds")
        index = self.indices[index]

        if self.stage == 'test':
            return self.__gettest__(index)
        return self.__getothers(index)

    def __gettest__(self, index):
        """Generate one batch of test data"""

        # Generate data
        sample, metadata = self.__test_generation(index)
        if self.transform:
            sample = self.transform(sample)

        return sample, metadata

    @staticmethod
    def get_data(file_path):
        """
        Given a file path, loads test file (in h5 format).
        Returns: tensor of shape (number_of_test_cases = 288, 496, 435, 3)
        """
        with h5py.File(file_path if isinstance(file_path, str) else str(file_path), "r") as fr:
            data = fr.get("array")
            return data[()]  # np.array(data)

    @staticmethod
    def write_data(data_in, file_path):
        """
        write data in gzipped h5 format.
        """
        with h5py.File(file_path, 'w', libver='latest') as f:
            f.create_dataset('array',
                             shape=data_in.shape,
                             data=data_in,
                             chunks=(1, *data_in.shape[1:]),
                             compression='gzip',
                             dtype='uint8',
                             compression_opts=9)

    @staticmethod
    def process_output(data, n_out_channel=None, collapse_time=True):
        n_out_channel = n_out_channel or 8  # self.n_out_channel
        x = data.cpu().numpy() * 255.0
        x_shape = x.shape
        if collapse_time:
            return x.reshape(x_shape[0], -1, n_out_channel,
                             *x_shape[2:]).transpose(0, 1, 3, 4, 2).astype(np.uint8)  # right mapping
        return x.transpose(0, 1, 3, 4, 2).astype(np.uint8)  # correct version
        # return x.transpose(0, 2, 3, 4, 1).astype(np.uint8)  # wrong version

    @staticmethod
    def process_input(data, collapse_time=True):
        if collapse_time:
            d_shape = data.shape
            return data.transpose(1, 2, 0, 3).reshape(*d_shape[1:3], -1)  # right mapping
        return data

    def get_terms(self, index):
        file_index = index // self.parts_per_file
        start_index = index % self.parts_per_file

        start_index = start_index * self.sampling_step  # get actual location

        file_path = self.files[file_index]
        file_name = basename(str(file_path))

        if self.file_index != file_index:
            self.file_index = file_index
            if self.stage == 'test':
                self.data_additional = self.get_data(file_path)
                self.data = self.get_data(str(file_path).replace('_additional', ''))
                # city = file_name.split('_')[0]
            else:  # training
                self.data = self.get_data(file_path)
                # city = file_name.split('_')[1]

        if self.stage == 'test':
            date_data = self.data_additional[start_index]
            metadata = {'lead_time': date_data[1], 'week_day': date_data[0], 'part_index': start_index,
                        'file_name': file_name.replace('_additional', '')}
        else:  # training
            date = file_name.split('_')[0]
            date_code = time.strptime(str(date), "%Y-%m-%d")
            metadata = {'lead_time': min(start_index, 255), 'week_day': date_code.tm_wday,
                        'part_index': start_index, 'file_name': file_name}
        metadata['mask'] = self.static_mask
        return start_index, metadata

    def get_extra(self, x, metadata):
        if self.use_static:
            # x_static = self.static_data
            if self.collapse_time:
                x_static = self.static_map
            else:
                x_static = repeat(self.static_map, 'h w d -> c h w d', c=self.n_frame_in)
            x = np.concatenate([x, x_static], axis=-1)

        if self.use_time_slot:
            x_time = [metadata[label] * np.ones_like(x[..., :1]) for label in ['lead_time', 'week_day']]
            x = np.concatenate([x, *x_time], axis=-1)

        return x

    def __data_generation(self, index):
        """generate training data"""
        start_index, metadata = self.get_terms(index)
        # Store sample
        mid_index = start_index + self.n_frame_in
        end_index = [mid_index + t // 5 - 1 for t in self.times_out]

        x = self.process_input(self.data[start_index:mid_index, :, :, :], self.collapse_time)
        y = self.process_input(self.data[end_index, :, :, :self.n_out_channel], self.collapse_time)

        x = self.get_extra(x, metadata)
        return (x, y), metadata

    def __test_generation(self, index):
        """generate training data"""

        start_index, metadata = self.get_terms(index)
        # x = self.process_input(self.data[start_index])
        x = self.process_input(self.data[start_index], self.collapse_time)

        x = self.get_extra(x, metadata)

        return (x,), metadata


# To split the datasets
def split_dataset(dataset, lengths, seed=0):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
        seed: manual seed for reproducibility
    """
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=torch.Generator().manual_seed(seed)).tolist()
    return [torch.utils.data.Subset(dataset, indices[offset - length: offset])
            for offset, length in zip(_accumulate(lengths), lengths)]
