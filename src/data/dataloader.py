import torch
import numpy as np
import torchvision.transforms as vtransforms

from collections import Counter
from torch.nn.functional import one_hot
from torch.utils.data import random_split
from torch.utils.data.sampler import WeightedRandomSampler

import src.utils.signal_processing as sp

classes = {
    "Task_11": ["Normal", "Adventitious"],
    "Task_12": [
        "Normal",
        "Rhonchi",
        "Wheeze",
        "Stridor",
        "Coarse Crackle",
        "Fine Crackle",
        "Wheeze&Crackle",
    ],
    "Task_21": ["Normal", "Poor Quality", "Adventitious"],
    "Task_22": ["Normal", "Poor Quality", "CAS", "DAS", "CAS & DAS"],
}


def preprocessing(
    audio, og_sr, max_length, feature, pos_norm="zscore", new_sr=None, **kwargs
):
    if new_sr is not None:
        audio = sp.resampling(audio, og_sr, new_sr)
        sr_in = new_sr
    else:
        sr_in = og_sr
    max_width = int(max_length * sr_in)
    if "hop_length" in kwargs:
        max_width = int(max_width / kwargs.get("hop_length"))
    # feature = sp.MFCC(audio=audio, sampling_rate=sr_in, **kwargs)
    feature = sp.featureSelector(
        signal=audio, sr=sr_in, feature=feature, pos_norm=pos_norm, **kwargs
    )
    resize_feature = vtransforms.RandomCrop(
        size=(feature.shape[1], max_width), padding=0, pad_if_needed=True
    )(torch.from_numpy(feature).float())
    return feature, resize_feature


def label2idx(word, task):
    """Label to index.
    Args:
        word: actual label.
        task: full task.
    Returns:
        index of label.
    """
    return torch.tensor(classes[task].index(word))


def idx2label(index, task):
    """Index to label.
    Args:
        index: index.
        task: full task.
    Returns:
        label.
    """
    return classes[task][index]


def supcon_collate(batch, full_task, sub_task, transform):
    tensors, targets, infos = [], [], []

    for feature, label, info in batch:
        feature = torch.from_numpy(feature).float()
        tensors.append(transform(feature))

        label = label[sub_task - 1]
        targets += [label2idx(label, full_task)]
        infos += [one_hot(torch.tensor(info[-1]) - 1, num_classes=4).float()]       
    
    x_values, y_values = zip(*tensors)
    tensors = [torch.stack(x_values), torch.stack(y_values)]

    targets = torch.stack(targets)
    infos = torch.stack(infos)

    return tensors, targets, infos


def mfcc_collate(
    batch, og_sr, max_length, full_task, sub_task, pos_norm="zscore", **kwargs
):
    tensors, targets, infos = [], [], []

    # Gather in lists, and encode labels as indices
    for feature, label, info in batch:
        _, processed = preprocessing(
            feature, og_sr, max_length, feature="mfcc", pos_norm=pos_norm, **kwargs
        )
        tensors += [processed]
        label = label[sub_task - 1]
        targets += [label2idx(label, full_task)]
        infos += [one_hot(torch.tensor(info[-1]) - 1, num_classes=4).float()]

    # Group the list of tensors into a batched tensor
    tensors = torch.stack(tensors)
    targets = torch.stack(targets)
    infos = torch.stack(infos)

    # targets.float().unsqueeze(1)
    return tensors, targets, infos


def custom_collate(batch, max_length, full_task, sub_task):
    """Customized collate function.
    Args:
        batch: batch size.
        max_length: the maximum length of each frame.
        full_task: the full name of task.
        sub_task: the sub task number (1 or 2).
    Returns:
        Tuple of (tensors, targets, infos).
    """
    tensors, targets, infos = [], [], []

    # Gather in lists, and encode labels as indices
    for feature, label, info in batch:
        feature = torch.from_numpy(feature).float()
        size = feature.shape
        tensors += [
            vtransforms.RandomCrop(
                size=(size[1], max_length), padding=0, pad_if_needed=True
            )(feature)
        ]
        label = label[sub_task - 1]
        targets += [label2idx(label, full_task)]
        infos += [one_hot(torch.tensor(info[-1]) - 1, num_classes=4).float()]

    # Group the list of tensors into a batched tensor
    tensors = torch.stack(tensors)
    targets = torch.stack(targets)
    infos = torch.stack(infos)

    # targets.float().unsqueeze(1)
    return tensors, targets, infos


def trainValLoader(
    dataset,
    sub_task,
    collate_fn,
    valid_size=0.1,
    batch_size=32,
    train_sampler="balanced",
    val_sampler="none",
    num_workers=0,
):
    """Generate Train and validation Dataloaders.
    Args:
        dataset: A dataset pending to split into train and valid.
        sub_task: sub task in number (1 or 2).
        collate_fn: collate function.
        valid_size: partition for validation in input "dataset".
        batch_size: batch size.
        train_sampler: method of sampler in traindataset (default is "balanced").
        num_workers: parallel of processing if not 0 (default is 0).
    Returns:
        dictionary includes trainloader and validloader.
        {"train": trainloader, "val":validloader}
    """
    print(
        "Getting Data... {}% Validation Set\n".format(int(np.around(valid_size * 100)))
    )
    data_size = len(dataset)
    split = int(np.floor(valid_size * data_size))
    trainDataset, validDataset = random_split(dataset, [data_size - split, split])

    print("Batch Size:", batch_size)
    print("\nTrain Len =", len(trainDataset), ", Validation Len =", len(validDataset))

    if train_sampler == "balanced":
        print("\nBalanced sampler is used for train.")
        train_indices = trainDataset.indices
        y_train = [dataset[i][1][sub_task - 1] for i in train_indices]
        class_sample_count = Counter(y_train)
        print("Number of samples in each class:\n", class_sample_count)
        weight = class_sample_count.copy()

        for key in class_sample_count.keys():
            weight[key] = 1.0 / class_sample_count.get(key)

        samples_weight = np.array([weight[t] for t in y_train])
        samples_weight = torch.from_numpy(samples_weight)

        balance_sampler = WeightedRandomSampler(
            samples_weight.double(), len(samples_weight)
        )

        trainloader = torch.utils.data.DataLoader(
            trainDataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=balance_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
    else:
        print("\nBalanced sampler is not used for train.")
        trainloader = torch.utils.data.DataLoader(
            trainDataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    if val_sampler == "balanced":
        print("\nBalanced sampler is used for val.")
        val_indices = validDataset.indices
        y_val = [dataset[i][1][sub_task - 1] for i in val_indices]
        class_sample_count = Counter(y_val)
        print("Number of samples in each class:\n", class_sample_count)
        weight = class_sample_count.copy()

        for key in class_sample_count.keys():
            weight[key] = 1.0 / class_sample_count.get(key)

        samples_weight = np.array([weight[t] for t in y_val])
        samples_weight = torch.from_numpy(samples_weight)

        balance_sampler = WeightedRandomSampler(
            samples_weight.double(), len(samples_weight)
        )

        validloader = torch.utils.data.DataLoader(
            validDataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=balance_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
    else:
        print("\nBalanced sampler is not used for val.")
        validloader = torch.utils.data.DataLoader(
            validDataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
    
    print("")
    print(
        "Train Size Batched =",
        int(len(trainloader.dataset) / batch_size),
        ", Validation Size Batched =",
        int(len(validloader.dataset) / batch_size),
    )

    return {"train": trainloader, "val": validloader}


def testLoader(
    testDataset,
    collate_fn,
    batch_size=32,
    num_workers=0,
    shuffle_in=False,
):
    """Generate testloader.
    Args:
        testDataset: dataset for test.
        collate_fn: collate function.
        batch_size: batch size.
        num_workers: parallel of processing if not 0 (default is 0).
        shuffle_in: True to have the data reshuffled at every epoch. (default is False).
    Returns:
        testloader
    """
    print("Batch Size:", batch_size)
    print("Test Len =", len(testDataset), end=", ")

    testloader = torch.utils.data.DataLoader(
        testDataset,
        batch_size=batch_size,
        shuffle=shuffle_in,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    print("Test Size Batched =", int(len(testloader.dataset) / batch_size))

    return testloader
