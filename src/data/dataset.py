import json
import torchaudio
import numpy as np

from os import listdir
from os.path import join
from torch.utils.data import Dataset

import src.utils.signal_processing as sp


def genDatasetsv2(
    task,
    data_dict,
    **kwargs,
):
    """Generate datasets in a data path dictionary.
    Args:
        task: main task of project (1 or 2).
        data_dict: data path dictionary (i.e. {"train": ['wav_dir', 'json_dir']}).
        resample: resample audio in new sampling rate (default is None).
        feature: do feature extraction (default is True).
        pos_norm: post normalization method (default is zscore).
        **kwargs: other parameters send to signal processing.
    Returns:
        A tuple of Datasets (length should be same as of data_dict).
    """
    PRSDataset = {}
    for i in data_dict:
        dataset = subDatasetsv2(
            task=task,
            wav_dir=data_dict[i][0],
            json_dir=data_dict[i][1],
            **kwargs,
        )
        PRSDataset.update({i: dataset})

    return [PRSDataset[i] for i in data_dict]


def genDatasets(
    task,
    data_dict,
    resample=None,
    feature=True,
    pos_norm="zscore",
    **kwargs,
):
    """Generate datasets in a data path dictionary.
    Args:
        task: main task of project (1 or 2).
        data_dict: data path dictionary (i.e. {"train": ['wav_dir', 'json_dir']}).
        resample: resample audio in new sampling rate (default is None).
        feature: do feature extraction (default is True).
        pos_norm: post normalization method (default is zscore).
        **kwargs: other parameters send to signal processing.
    Returns:
        A tuple of Datasets (length should be same as of data_dict).
    """
    PRSDataset = {}
    for i in data_dict:
        dataset = trainDatasets(
            task=task,
            wav_dir=data_dict[i][0],
            json_dir=data_dict[i][1],
            resample=resample,
            feature=feature,
            pos_norm=pos_norm,
            **kwargs,
        )
        PRSDataset.update({i: dataset})

    return [PRSDataset[i] for i in data_dict]


class subDatasetsv2(Dataset):
    """Create datasets for train process, whose raw data includes labels.
    Args:
        task: main task of project (1 or 2).
        wav_dir: data path for wav files.
        json_dir: data path for json files.
        resample: resample audio in new sampling rate (default is None).
        feature: do feature extraction (default is True).
        pos_norm: post normalization method (default is zscore).
        **kwargs: other parameters send to signal processing.
    Returns:
        Dataset includes (features, traget, info).
    """

    def __init__(
        self,
        task,
        wav_dir,
        json_dir,
        resample=None,
        **kwargs,
    ):
        assert task in [1, 2], "Invalid task chosen"
        isolated_prs = isolate_prs(task, wav_dir, json_dir, resample)
        self.isolated_prs = isolated_prs
        self.task = task
        self.kwargs = kwargs

    def __len__(self):
        return len(self.isolated_prs)

    def __getitem__(self, index):
        target = self.isolated_prs[index]["labels"]
        info = self.isolated_prs[index]["info"]

        sr = info[0]
        audio = np.array(self.isolated_prs[index]["signal"], dtype=float)
        return audio, tuple(target), tuple(info)


class trainDatasets(Dataset):
    """Create datasets for train process, whose raw data includes labels.
    Args:
        task: main task of project (1 or 2).
        wav_dir: data path for wav files.
        json_dir: data path for json files.
        resample: resample audio in new sampling rate (default is None).
        feature: do feature extraction (default is True).
        pos_norm: post normalization method (default is zscore).
        **kwargs: other parameters send to signal processing.
    Returns:
        Dataset includes (features, traget, info).
    """

    def __init__(
        self,
        task,
        wav_dir,
        json_dir,
        resample=None,
        feature=None,
        pre_emph=False,
        pos_norm="zscore",
        **kwargs,
    ):
        assert task in [1, 2], "Invalid task chosen"
        isolated_prs = isolate_prs(task, wav_dir, json_dir, resample)
        self.isolated_prs = isolated_prs
        self.task = task
        self.feature = feature
        self.pre_emph = pre_emph
        self.pos_norm = pos_norm
        self.kwargs = kwargs

    def __len__(self):
        return len(self.isolated_prs)

    def __getitem__(self, index):
        target = self.isolated_prs[index]["labels"]
        info = self.isolated_prs[index]["info"]

        sr = info[0]
        audio = np.array(self.isolated_prs[index]["signal"], dtype=float)
        if self.feature != None:
            features = sp.featureSelector(
                audio,
                sr,
                self.feature,
                pre_emph=self.pre_emph,
                pos_norm=self.pos_norm,
                **self.kwargs,
            )
        else:
            features = audio

        return features, tuple(target), tuple(info)


def isolate_prs(task, wav_dir, json_dir, resample):
    """Isolate PRS event based on start and end in json file.
    Args:
        task: main task of project (1 or 2).
        wav_dir: data path for wav files.
        json_dir: data path for json files.
        resample: resample audio in new sampling rate (default is None).
    Returns:
        isolated_prs: list of dict. i.e.
            {
                "signal": audio,
                "info": [sr, patiend_id, rec_id, age, gender, loc],
                "labels": [label_11, label_12],  # if task is 1
            }
    """
    assert task in [1, 2], "Invalid task chosen"

    isolated_prs = []
    for recording in listdir(json_dir):
        name = recording[:-5]
        wav_name = name + ".wav"
        entry = name.split("_")
        patiend_id = int(entry[0])
        age = float(entry[1])
        gender = int(entry[2])
        loc = int(entry[3][-1])
        rec_id = int(entry[4])

        with open(join(json_dir, recording)) as f:
            rec_json = json.load(f)

        audio, sr = torchaudio.load(join(wav_dir, wav_name))

        if task == 1:
            events = rec_json["event_annotation"]
            clip_prs = []
            for i, event in enumerate(events):
                label_12 = event["type"].replace("+", "&")
                label_11 = "Adventitious" if label_12 != "Normal" else label_12

                start = int(int(event["start"]) / 1000 * sr)
                end = int(int(event["end"]) / 1000 * sr)
                clip_audio = audio[:, start:end].numpy()
                if resample is not None:
                    clip_audio, sr = sp.resampling(clip_audio, sr, resample)
                clip_prs.append(
                    {
                        "signal": clip_audio,
                        "info": [sr, patiend_id, rec_id, age, gender, loc],
                        "labels": [label_11, label_12],
                    }
                )
            isolated_prs.extend(clip_prs)

        elif task == 2:
            label_22 = rec_json["record_annotation"]
            label_21 = (
                "Adventitious"
                if label_22 not in ("Normal", "Poor Quality")
                else label_22
            )
            audio = audio.numpy()
            if resample is not None:
                audio, sr = sp.resampling(audio, sr, resample)
            isolated_prs.append(
                {
                    "signal": audio,
                    "info": [sr, patiend_id, rec_id, age, gender, loc],
                    "labels": [label_21, label_22],
                }
            )

    return isolated_prs
