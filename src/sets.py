import numpy as np

import torch
from torchvision.transforms import v2

from datasets import (
    load_dataset,
    concatenate_datasets,
    Dataset,
    IterableDataset,
    DatasetDict,
)

from sklearn.model_selection import StratifiedKFold

SEED = 42
COLS = ["pixel_values", "labels"]


# This is better than defining Torch transforms in the function, where a new
# transform object will be instantiated every time.
def get_transform(train: bool):
    train_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.RandomPosterize(bits=6),
            v2.ColorJitter(brightness=0.4, contrast=0.1, saturation=0.1, hue=0.1),
            v2.RandomRotation(degrees=(0, 180)),
            v2.GaussianBlur(kernel_size=5, sigma=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(224, 224), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def transform(sample):
        if train:
            sample["pixel_values"] = train_transforms(sample["pixel_values"])
        else:
            sample["pixel_values"] = test_transforms(sample["pixel_values"])

        return sample

    return transform


def sample_dataset(ds: Dataset, repeat: float = 1, seed=SEED):
    rep_int = int(repeat)
    rep_frac = repeat - rep_int
    rep_frac_num = int(rep_frac * len(ds))


    final = ds.shuffle(seed=seed).select(range(rep_frac_num))
    for i in range(rep_int):
        final = concatenate_datasets([final, ds])

    # print(repeat, rep_int, rep_frac)
    # print(len(ds), len(final))
    return final


def resample_populations(
    ds: Dataset, neg_pos_ratio: float = 1, pos_repeat: float = 1, seed=SEED
):
    pos = ds.filter(lambda label: label == 1, input_columns="labels")
    neg = ds.filter(lambda label: label == 0, input_columns="labels")

    # upsampled positive set
    pos = sample_dataset(pos, pos_repeat)
    num_pos = len(pos)

    num_neg = int(num_pos * neg_pos_ratio)
    neg = sample_dataset(neg, num_neg / len(neg))

    return concatenate_datasets([pos, neg]).shuffle(seed=seed)


def get_dataset_splits(ds_source):
    ds: DatasetDict = load_dataset(ds_source)

    ds = ds.rename_column("image", "pixel_values")
    ds = ds.rename_column("label", "labels")

    return ds["train"], ds["validation"]

def get_source_splits(ds):
    ds_cp = ds.filter(
        lambda source: source == "complete", input_columns="PROC_source"
    )
    ds_24 = ds.filter(
        lambda source: source == "2024", input_columns="PROC_source"
    )

    return ds_cp, ds_24


def finalise_splits(ds, train:bool, nproc, cols=COLS, seed=SEED):
    ds = (
        ds.to_iterable_dataset(num_shards=nproc)
        .map(get_transform(train))
        .shuffle(seed=seed)
        .select_columns(cols)
    )

    return ds


def _prepare_2024_resampled(samp: tuple[float, float], nproc: int = 16):
    """
    Load from deprecated 2024 repo:

    - Way faster download but no metadata
    - No guarantee of same dataset splits
    - No guarantee of same deterministic shuffling as that from full set subset.

    Use for TESTING only.
    """

    train, val = get_dataset_splits("TobanDjan/isic_2024")
    train = resample_populations(train, *samp)

    train = finalise_splits(train, True, nproc)
    val = finalise_splits(val, False, nproc)
    
    return train, val


def prepare_2024_resampled(samp: tuple[float, float], nproc: int = 16):
    train, val = get_dataset_splits("TobanDjan/isic_full")

    source_filter = lambda source: source == "2024"
    train = train.filter(source_filter, input_columns="PROC_source")
    val = val.filter(source_filter, input_columns="PROC_source")

    train = resample_populations(train, *samp)

    train = finalise_splits(train, True, nproc)
    val = finalise_splits(val, False, nproc)
    
    return train, val


def prepare_full_resampled(
    samp_cp: tuple[float, float], samp_24: tuple[float, float], nproc: int = 16
):
    """
    Sampling tuples are passed into the resample function.
    """

    train, val = get_dataset_splits("TobanDjan/isic_full")

    train_cp, train_24 = get_source_splits(train)
    # print(len(train.filter(lambda row: row['labels'] == 1)))
    # print(len(train.filter(lambda row: row['labels'] == 0)))

    train_cp = resample_populations(train_cp, *samp_cp)
    train_24 = resample_populations(train_24, *samp_24)
    train = concatenate_datasets([train_cp, train_24])

    train = finalise_splits(train, True, nproc)
    val = finalise_splits(val, False, nproc)
    
    return train, val


# saves half of the boring-ass boilerplate
def _prepare_kfold(ds:Dataset,
                   samp:tuple[float, float],
                   n_splits:int,
                   shuffle:bool = True,
                   seed:int = SEED,
                   nproc:int=16):
    ds = resample_populations(ds, *samp)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    for train_idx, val_idx in skf.split(np.zeros(ds.num_rows), ds['labels']):
        train = ds.select(train_idx)  
        val = ds.select(val_idx)

        yield train, val


def prepare_full_kfold(samp_cp, samp_24, n_splits, shuffle=True, seed=SEED,
                       nproc=16):
    train, val = get_dataset_splits("TobanDjan/isic_full") 

    comb = concatenate_datasets([train, val])
    comb_cp, comb_24 = get_source_splits(comb)

    # I know it's ugly as hell, but seems to be the quickest way around...
    for split_cp, split_24 in zip(
            _prepare_kfold(comb_cp, samp_cp, n_splits, shuffle, seed, nproc),
            _prepare_kfold(comb_24, samp_24, n_splits, shuffle, seed, nproc)
            ):
        train_cp, val_cp = split_cp  
        train_24, val_24 = split_24

        train = concatenate_datasets([train_cp, train_24])
        val = concatenate_datasets([val_cp, val_24])
    
        train = finalise_splits(train, True, nproc, seed=seed)
        val = finalise_splits(val, False, nproc, seed=seed)

        yield train, val


def prepare_2024_kfold(n_splits, samp, shuffle=True, seed=SEED, nproc=16):
    train, val = get_dataset_splits("TobanDjan/isic_full") 

    comb = concatenate_datasets([train, val])
    _, comb_24 = get_source_splits(comb)

    for train, val in _prepare_kfold(comb_24, samp, n_splits, shuffle, seed, nproc):
        # pos = train.filter(lambda label: label == 1, input_columns="labels")
        # neg = train.filter(lambda label: label == 0, input_columns="labels")
        # print(len(pos), len(neg))

        train = finalise_splits(train, True, nproc, seed=seed)
        val = finalise_splits(val, False, nproc, seed=seed)

        yield train, val
