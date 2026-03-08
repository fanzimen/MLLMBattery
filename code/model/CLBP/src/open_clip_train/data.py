import ast
import json
import logging
import math
import os
import random
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
# --- [新增] --- 引入分布式工具
from open_clip_train.distributed import is_master, broadcast_object
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample
from typing import Optional, List, Tuple
import glob
from bisect import bisect_right
from pathlib import Path
try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts

class TimeSeriesDataset(Dataset):
    def __init__(self, input_filename, description_filename, window_size=20, tokenizer=None, transforms=None, mean=None, std=None):
        """
        初始化时序数据集
        :param input_filename: 包含时序特征的CSV文件路径
        :param description_filename: 包含文本描述的CSV文件路径
        :param window_size: 滑动窗口大小
        :param tokenizer: 文本分词器
        :param transforms: 应用于时序数据的额外转换（可选）
        :param mean: 整个训练集的特征均值
        :param std: 整个训练集的特征标准差
        """
        logging.debug(f'Loading time series data from {input_filename}.')
        # 加载时序数据和文本描述
        self.data_df = pd.read_csv(input_filename)
        self.desc_df = pd.read_csv(description_filename)
        
        # 提取数值特征列,去掉时序数据的'soh'列和'capacity'列（如果存在）
        self.feature_columns = self.data_df.columns.drop(['soh', 'capacity'], errors='ignore')
        self.features = self.data_df[self.feature_columns].values
        
        self.window_size = window_size
        self.transforms = transforms
        self.tokenize = tokenizer

        # --- [核心修改] ---
        # 存储全局的均值和标准差
        if mean is not None and std is not None:
            self.mean = torch.tensor(mean, dtype=torch.float32)
            self.std = torch.tensor(std, dtype=torch.float32)
            # 防止除以零
            self.std[self.std == 0] = 1.0
        else:
            self.mean = None
            self.std = None
        
        logging.debug('Done loading data.')

    def __len__(self):
        # 样本数量为总周期数减去窗口大小再加1
        return len(self.features) - self.window_size + 1

    def __getitem__(self, idx):
        # 定义滑窗的起始和结束位置
        start_idx = idx
        end_idx = idx + self.window_size
        
        # 提取窗口数据
        window_features = self.features[start_idx:end_idx]
        
        # 转换为Tensor
        ts_data = torch.FloatTensor(window_features)

        # --- [核心修改] ---
        # 使用全局的均值和标准差进行标准化
        if self.mean is not None and self.std is not None:
            ts_data = (ts_data - self.mean) / self.std

        if self.transforms:
            ts_data = self.transforms(ts_data)
            
        # 获取滑窗最后一个cycle对应的文本描述
        # 文本描述文件中的索引对应cycle-1，滑窗最后一个cycle的索引是end_idx - 1
        caption_idx = end_idx - 1
        caption = self.desc_df.loc[caption_idx, 'description']
        
        # 对文本进行分词
        text_data = self.tokenize([str(caption)])[0]
        
        return ts_data, text_data


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights),\
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0 

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."
    
    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train)
    ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)







class SyntheticDataset(Dataset):

    def __init__(
            self,
            transform=None,
            image_size=(224, 224),
            caption="Dummy caption",
            dataset_size=100,
            tokenizer=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    if dataset_type == "timeseries":
        return get_timeseries_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns, epoch=0, model_type="timeseries",tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)



    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")


    return data

# def get_timeseries_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
#     input_filename = args.train_data if is_train else args.val_data
#     description_filename = args.train_description_data if is_train else args.val_description_data
#     assert input_filename
#     assert description_filename

#     # --- [核心修改] ---
#     # 1. 计算或加载全局的均值和标准差
#     # 这个统计必须只在训练数据上计算
#     train_df = pd.read_csv(args.train_data)
#     feature_columns = train_df.columns.drop(['soh', 'capacity'], errors='ignore')
#     mean = train_df[feature_columns].mean().values
#     std = train_df[feature_columns].std().values

#     dataset = TimeSeriesDataset(
#         input_filename,
#         description_filename,
#         window_size=args.window_size,
#         transforms=preprocess_fn,
#         tokenizer=tokenizer,
#         mean=mean,  # 传递均值
#         std=std     # 传递标准差
#     )
#     num_samples = len(dataset)
#     sampler = DistributedSampler(dataset) if args.distributed and is_train else None
#     shuffle = is_train and sampler is None

#     dataloader = DataLoader(
#         dataset,
#         batch_size=args.batch_size,
#         shuffle=shuffle,
#         num_workers=args.workers,
#         pin_memory=True,
#         sampler=sampler,
#         drop_last=is_train,
#     )
#     dataloader.num_samples = num_samples
#     dataloader.num_batches = len(dataloader)

#     return DataInfo(dataloader, sampler)

class TimeSeriesDataset(Dataset):
    # 保留以兼容旧的单文件用法
    def __init__(self, input_filename, description_filename, window_size=20, tokenizer=None, transforms=None, mean=None, std=None):
        """
        初始化时序数据集
        :param input_filename: 包含时序特征的CSV文件路径
        :param description_filename: 包含文本描述的CSV文件路径
        :param window_size: 滑动窗口大小
        :param tokenizer: 文本分词器
        :param transforms: 应用于时序数据的额外转换（可选）
        :param mean: 整个训练集的特征均值
        :param std: 整个训练集的特征标准差
        """
        logging.debug(f'Loading time series data from {input_filename}.')
        # 加载时序数据和文本描述
        self.data_df = pd.read_csv(input_filename)
        self.desc_df = pd.read_csv(description_filename)
        
        # 提取数值特征列,去掉时序数据的'soh'列和'capacity'列（如果存在）
        self.feature_columns = self.data_df.columns.drop(['soh', 'capacity'], errors='ignore')
        self.features = self.data_df[self.feature_columns].values
        
        self.window_size = window_size
        self.transforms = transforms
        self.tokenize = tokenizer

        # --- [核心修改] ---
        # 存储全局的均值和标准差
        if mean is not None and std is not None:
            self.mean = torch.tensor(mean, dtype=torch.float32)
            self.std = torch.tensor(std, dtype=torch.float32)
            # 防止除以零
            self.std[self.std == 0] = 1.0
        else:
            self.mean = None
            self.std = None
        
        logging.debug('Done loading data.')

    def __len__(self):
        # 样本数量为总周期数减去窗口大小再加1
        return len(self.features) - self.window_size + 1

    def __getitem__(self, idx):
        # 定义滑窗的起始和结束位置
        start_idx = idx
        end_idx = idx + self.window_size
        
        # 提取窗口数据
        window_features = self.features[start_idx:end_idx]
        
        # 转换为Tensor
        ts_data = torch.FloatTensor(window_features)

        # --- [核心修改] ---
        # 使用全局的均值和标准差进行标准化
        if self.mean is not None and self.std is not None:
            ts_data = (ts_data - self.mean) / self.std

        if self.transforms:
            ts_data = self.transforms(ts_data)
            
        # 获取滑窗最后一个cycle对应的文本描述
        # 文本描述文件中的索引对应cycle-1，滑窗最后一个cycle的索引是end_idx - 1
        caption_idx = end_idx - 1
        caption = self.desc_df.loc[caption_idx, 'description']
        
        # 对文本进行分词
        text_data = self.tokenize([str(caption)])[0]
        
        return ts_data, text_data
    
# 新增：清洗函数（处理 -inf/inf/NaN）
def _clean_features(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    arr = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if not finite.all():
        for c in range(arr.shape[1]):
            col = arr[:, c]
            mask = np.isfinite(col)
            mean_val = col[mask].mean() if mask.any() else 0.0
            col[~mask] = mean_val
            arr[:, c] = col
    return arr

# 新增：列出目录下 CSV 文件（不递归）
def _list_csvs(path_or_file: str) -> list[str]:
    p = Path(path_or_file)
    if p.is_dir():
        return sorted([str(x) for x in p.glob("*.csv")])
    elif p.is_file():
        return [str(p)]
    else:
        raise FileNotFoundError(f"Path not found: {path_or_file}")

# 新增：根据 soh 文件名在描述根目录找到同名描述文件
def _resolve_desc_path(soh_csv: str, desc_root: str) -> str:
    basename = Path(soh_csv).name
    candidate = Path(desc_root) / basename
    if not candidate.is_file():
        logging.warning(f"Description CSV not found for {soh_csv} under {desc_root}. Using empty descriptions.")
    return str(candidate)

# 新增：从 soh 目录推导默认描述目录 data_preprocess/full_descriptions
def _infer_default_desc_root(path_or_file: str) -> str:
    # 假设结构为 .../data_preprocess/soh_data/{train_x|test_x}
    p = Path(path_or_file)
    base = p if p.is_dir() else p.parent
    # 上两级到 data_preprocess
    dp = base.parent.parent if base.parent.parent is not None else base
    return str(dp / "full_descriptions")


# 新增：多文件时序数据集
class MultiFileTimeSeriesDataset(Dataset):
    def __init__(
        self,
        soh_files: list[str],
        desc_root: Optional[str],
        window_size: int = 20,
        tokenizer=None,
        transforms=None,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        self.window_size = window_size
        self.transforms = transforms
        self.tokenize = tokenizer
        # empty_desc = 0
        # total_desc = 0
        # 读取第一个文件确定特征列（排除 soh/capacity）
        if len(soh_files) == 0:
            raise ValueError("No SOH csv files found.")
        first_df = pd.read_csv(soh_files[0])
        self.feature_columns = first_df.columns.drop(['soh', 'capacity'], errors='ignore').tolist()

        # 解析描述根目录
        if desc_root is None:
            desc_root = _infer_default_desc_root(soh_files[0])
        self.desc_root = desc_root

        # 预加载所有文件的数据与描述文本
        self._features_per_file: list[np.ndarray] = []
        self._descs_per_file: list[list[str]] = []
        self._sizes: list[int] = []  # 每文件可用样本（滑窗个数）
        total = 0
        for soh in soh_files:
            try:
                df = pd.read_csv(soh)
            except Exception as e:
                logging.warning(f"Read csv failed for {soh}, skipping. Err: {e}")
                continue
            feats = _clean_features(df, self.feature_columns)
            # 加载描述
            desc_csv = _resolve_desc_path(soh, self.desc_root)
            desc_list = []
            if Path(desc_csv).is_file():
                try:
                    desc_df = pd.read_csv(desc_csv)
                    # 兼容：若没有 description 列则用空字符串
                    if 'description' in desc_df.columns:
                        desc_list = desc_df['description'].astype(str).tolist()
                    else:
                        desc_list = [""] * len(df)
                        logging.warning(f"No 'description' column in {desc_csv}, using empty strings.")
                except Exception as e:
                    logging.warning(f"Read desc csv failed for {desc_csv}, using empty strings. Err: {e}")
                    desc_list = [""] * len(df)
            else:
                desc_list = [""] * len(df)
            # # 统计空描述
            # total_desc += len(desc_list)
            # empty_desc += sum(1 for s in desc_list if not s.strip())

            n = feats.shape[0] - self.window_size + 1
            if n <= 0:
                logging.warning(f"File {soh} too short for window_size={self.window_size}, skipping.")
                continue

            self._features_per_file.append(feats)
            self._descs_per_file.append(desc_list)
            self._sizes.append(n)
            total += n

        if total == 0:
            raise ValueError("No valid windows after loading all files.")
        # if total_desc > 0:
        #     ratio = empty_desc / total_desc
        #     logging.warning(f"[TimeSeries] descriptions empty ratio = {ratio:.2%} "
        #                     f"(empty={empty_desc}, total={total_desc}). "
        #                     f"High empty ratio will hurt contrastive learning.")
        # 前缀和用于全局索引到具体文件映射
        self._cumulative = np.cumsum([0] + self._sizes).astype(int)

        # 保存标准化统计
        if mean is not None and std is not None:
            self.mean = torch.tensor(mean, dtype=torch.float32)
            self.std = torch.tensor(std, dtype=torch.float32)
            self.std[self.std == 0] = 1.0
        else:
            self.mean = None
            self.std = None

    def __len__(self):
        return int(self._cumulative[-1])

    def _locate(self, idx: int) -> tuple[int, int]:
        # 返回 (file_idx, local_start_idx)
        fidx = bisect_right(self._cumulative, idx) - 1
        local = idx - self._cumulative[fidx]
        return fidx, local

    def __getitem__(self, idx: int):
        fidx, start = self._locate(idx)
        feats = self._features_per_file[fidx]
        end = start + self.window_size
        window = torch.from_numpy(feats[start:end])  # (T, C)

        if self.mean is not None and self.std is not None:
            window = (window - self.mean) / self.std

        if self.transforms:
            window = self.transforms(window)

        # 描述文本取窗口末端对应索引
        desc_list = self._descs_per_file[fidx]
        cap_idx = min(end - 1, len(desc_list) - 1)
        caption = desc_list[cap_idx] if cap_idx >= 0 else ""

        text = self.tokenize([str(caption)])[0]
        return window, text

# 新增：统计训练集全局均值/标准差（流式，避免内存爆）
def _compute_global_mean_std(soh_files: list[str], feature_columns: list, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    # 这里按样本行（非滑窗）做标准化统计，和单文件版本一致
    count = 0
    running_sum = None
    running_sumsq = None
    for fp in soh_files:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            logging.warning(f"Read csv failed during stats for {fp}, skipping. Err: {e}")
            continue
        arr = _clean_features(df, feature_columns)
        if arr.size == 0:
            continue
        if running_sum is None:
            running_sum = np.zeros(arr.shape[1], dtype=np.float64)
            running_sumsq = np.zeros(arr.shape[1], dtype=np.float64)
        # 对每行加入统计
        finite = np.isfinite(arr)
        if not finite.all():
            # _clean_features 已处理，这里应全是有限
            pass
        running_sum += arr.sum(axis=0, dtype=np.float64)
        running_sumsq += np.square(arr, dtype=np.float64).sum(axis=0, dtype=np.float64)
        count += arr.shape[0]
    if count == 0:
        raise ValueError("Failed to compute mean/std: no valid data rows.")
    mean = (running_sum / count).astype(np.float32)
    var = (running_sumsq / count) - np.square(mean, dtype=np.float64)
    var = np.maximum(var, 0.0)
    std = np.sqrt(var).astype(np.float32)
    std[std == 0] = 1.0
    return mean, std

def get_timeseries_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_path = args.train_data if is_train else args.val_data
    assert input_path

    # 列出 SOH 文件
    soh_files = _list_csvs(input_path)
    if len(soh_files) == 0:
        raise ValueError(f"No CSV files found under: {input_path}")

    # 确定描述根目录优先级：--description-root > 旧参为目录 > 推导
    desc_root = getattr(args, 'description_root', None)
    fallback_desc = args.train_description_data if is_train else args.val_description_data
    if desc_root is None and fallback_desc is not None and Path(fallback_desc).is_dir():
        desc_root = fallback_desc
    if desc_root is None:
        desc_root = _infer_default_desc_root(input_path)

    # 计算或加载 global mean/std（仅基于训练集合并）
    if is_train:
        # 用首文件列来确定特征列
        first_df = pd.read_csv(soh_files[0])
        feature_columns = first_df.columns.drop(['soh', 'capacity'], errors='ignore').tolist()
        mean, std = _compute_global_mean_std(soh_files, feature_columns, args.window_size)
    else:
        # 验证/测试集重用训练集统计
        # 当 train_data 为目录时，同样按其目录下文件统计
        train_soh_files = _list_csvs(args.train_data)
        first_df = pd.read_csv(train_soh_files[0])
        feature_columns = first_df.columns.drop(['soh', 'capacity'], errors='ignore').tolist()
        mean, std = _compute_global_mean_std(train_soh_files, feature_columns, args.window_size)

    dataset = MultiFileTimeSeriesDataset(
        soh_files=soh_files,
        desc_root=desc_root,
        window_size=args.window_size,
        transforms=preprocess_fn,
        tokenizer=tokenizer,
        mean=mean,
        std=std
    )

    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)