### Setup
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as vtransforms
from torch.utils.data import Dataset, random_split
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import logging
import src.data.dataset as ds
import src.data.dataloader as dl
import src.test as customTest
import src.models.res_model as ResNet
import src.utils.metric as customMetric
from torch import nn, optim

import ray
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch

# -------------------- Globals --------------------#
# Device config
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {DEVICE} on {torch.cuda.get_device_name(0)} :D ")
MODEL_NAME = "ResNet_mfcc_tuning"
TASK_IN = "Task_11"
MAX_LENGTH = 3.0
SR = 8000
HOP_LENGTH = 256
N_MFCC = 64
N_FFT = 512
FEATURE = "mfcc"
VAL_PERCENT = 0.2
TRAIN_PERCENT = 1 - VAL_PERCENT
BATCH_SIZE = 32
LR = 0.001
EPOCH_NUM = 20
MAX_LENGTH_SAMPLES = int(MAX_LENGTH * SR / HOP_LENGTH)
N_F_BIN = 32  # Number of frequency bins, e.g. as number of MFCCs for MFCC
N_FFT = 512
INPUT_X_DIM = int(MAX_LENGTH * SR / HOP_LENGTH)  # Get the x dimension of a 2d plot
STRATEGY = ["score", "max"]
# Other options e.g. ["loss", "min"] or ["accuracy", "max"]

# TUNE_CONFIG = {
#     "c1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 3)),
#     "c2": tune.sample_from(lambda _: 2 ** np.random.randint(3, 5)),
#     "slen": tune.randint(2, 5),
#     "l1": tune.sample_from(lambda _: 2 ** np.random.randint(6, 9)),
#     "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 6)),
#     "lr": tune.loguniform(1e-4, 1e-1),
# }


TUNE_CONFIG = {
    "c1": tune.choice([2, 4, 6, 8]),
    "c2": tune.choice([8, 16, 32, 64]),
    "slen": tune.randint(2, 5),
    "l1": tune.choice([64, 128, 256, 512]),
    "l2": tune.choice([8, 16, 32, 64]),
    "lr": tune.loguniform(1e-4, 1e-1),
}

MFCC_CONFIG = {
    "hopLen": tune.quniform(64, 512, 32),
    "n_mfcc": tune.quniform(8, 64, 2),
    "lr": tune.loguniform(1e-4, 1e-1),
}

TUNE_MAX_EPOCH = 20
TUNE_GPU_PER_TRIAL = 1
TUNE_CPU_PER_TRIAL = 8
TUNE_SAMPLE_NUM = 10
# -------------------- End of Globals --------------------#


# -------------------- Logger function --------------------#
def setupLogger(name, logPath, level=logging.INFO):
    handler = logging.FileHandler(logPath)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


# -------------------- End of Logger function --------------------#


# -------------------- Configurable Network --------------------#
class configCNN(nn.Module):
    """An implementation of a lean CNN with just 2 convolutional layers and 3 fully-connected layers.
    Args:
       num_class: Number of class for classification.
    """

    def __init__(
        self, num_class, input_fdim, input_tdim, c1=6, c2=15, slen=3, l1=128, l2=64
    ):
        super().__init__()
        self.fdim = input_fdim
        self.tdim = input_tdim
        self.conv1 = nn.Conv2d(1, c1, slen)  # (6 ,30, 30)
        self.pool = nn.MaxPool2d(2, 2)  # (6, 15, 15)
        self.conv2 = nn.Conv2d(c1, c2, slen)  # (16, 13, 13)
        self.neurons = self.calc_flatten_fc_neuron()
        self.fc1 = nn.Linear(
            self.neurons, l1
        )  # (oiginal with mfcc - 2400, 128) (12000, 128 for melspec)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, num_class)
        self.softmax = nn.LogSoftmax(dim=1)

    def conv_block(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

    def calc_flatten_fc_neuron(self):
        size = self.conv_block(torch.randn(1, 1, self.fdim, self.tdim)).size()
        m = 1
        for i in size:
            m *= i
        return int(m)

    def forward(self, x):
        x = self.conv_block(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x


# --------------------End of Configurable Network --------------------#


# --------------------Tuning traning loop --------------------#
def tunePreprocess(config, dataset, num_classes):
    dataloader = dl.trainValLoader(
        dataset,
        sub_task,
        valid_size=VAL_PERCENT,
        batch_size=BATCH_SIZE,
        collate_fn=lambda batch: dl.mfcc_collate(
            batch,
            SR,
            MAX_LENGTH,
            TASK_IN,
            sub_task,
            pos_norm="zscore",
            hop_length=int(config["hopLen"]),
            n_mfcc=int(config["n_mfcc"]),
            n_fft=512,
        ),
        train_sampler="balanced",
    )
    tuneTrain(config, dataloader["train"], dataloader["val"], num_classes)


def tuneTrain(config, train_dl, val_dl, num_classes):
    network = ResNet.simpleResNet(num_classes=num_classes)
    network.to(DEVICE)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(params=network.parameters(), lr=config["lr"])
    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
        network.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(train_dl, 0):
            # get the inputs; data is a list of [inputs, labels, infod]
            inputs, labels, _ = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        truth = []
        preds = []
        for i, data in enumerate(val_dl, 0):
            with torch.no_grad():
                inputs, labels, _ = data
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = network(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                truth.append(labels.cpu().numpy().tolist())
                preds.append(predicted.cpu().numpy().tolist())
                preds_flat = [item for sublist in preds for item in sublist]
                truth_flat = [item for sublist in truth for item in sublist]
                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1
        val_score, *_ = customMetric.calc_score(
            truth_flat, preds_flat, False, int(TASK_IN[-2])
        )
        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and can be accessed through `session.get_checkpoint()`
        # API in future iterations.
        os.makedirs("tuning_models", exist_ok=True)
        torch.save(
            (network.state_dict(), optimizer.state_dict()),
            "tuning_models/checkpoint.pt",
        )
        checkpoint = Checkpoint.from_directory("tuning_models")
        session.report(
            {
                "loss": (val_loss / val_steps),
                "accuracy": correct / total,
                "score": val_score,
            },
            checkpoint=checkpoint,
        )
    print("Finished Training")


# --------------------End of Tuning traning loop --------------------#
if __name__ == "__main__":
    main_task = int(TASK_IN[-2])
    sub_task = int(TASK_IN[-1])
    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s:%(message)s")
    log_path = "logs/hyperTune.logs"
    data_path = "SPRSound"
    if not os.path.exists(log_path):
        open(log_path, "a").close()
    strat_target, strat_mode = STRATEGY
    tuneRecord = setupLogger("TuningLog", logPath=log_path, level=logging.INFO)

    data_dict = {
        "train": [
            os.path.join(data_path, "train_wav"),
            os.path.join(data_path, "train_json"),
        ],
        "intra_test": [
            os.path.join(data_path, "test_wav"),
            os.path.join(data_path, "test_json/intra_test_json"),
        ],
        "inter_test": [
            os.path.join(data_path, "test_wav"),
            os.path.join(data_path, "test_json/inter_test_json"),
        ],
    }
    trainDataset, intra_TestDataset, inter_TestDataset = ds.genDatasetsv2(
        task=main_task,
        data_dict=data_dict,
    )

    num_classes = len(dl.classes[TASK_IN])

    ## Edit Tuning parameters here
    scheduler = ASHAScheduler(max_t=TUNE_MAX_EPOCH, grace_period=1, reduction_factor=2)
    algo = HyperOptSearch(metric=strat_target, mode=strat_mode)
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                tunePreprocess, dataset=trainDataset, num_classes=num_classes
            ),
            resources={"cpu": TUNE_CPU_PER_TRIAL, "gpu": TUNE_GPU_PER_TRIAL},
        ),
        tune_config=tune.TuneConfig(
            metric=strat_target,
            mode=strat_mode,
            scheduler=scheduler,
            num_samples=TUNE_SAMPLE_NUM,
            search_alg=algo,
        ),
        param_space=MFCC_CONFIG,
    )

    results = tuner.fit()
    best_result = results.get_best_result(strat_target, strat_mode)
    best_val_loss = best_result.metrics["loss"]
    best_val_accu = best_result.metrics["accuracy"]
    best_val_score = best_result.metrics["score"]

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(best_val_loss))
    print("Best trial final validation accuracy: {}".format(best_val_accu))
    print("Best trial final validation score: {}".format(best_val_score))
    logMessage = (
        f"Model: {MODEL_NAME}, Task: {TASK_IN}, val_loss: {best_val_loss:>0.3}, "
        f"val_accuracy: {best_val_accu:>0.3}, best trial config: {best_result.config}"
    )
    tuneRecord.info(logMessage)

    ## Test the best Network
    inter_testloader = dl.testLoader(
        inter_TestDataset,
        batch_size=BATCH_SIZE,
        collate_fn=lambda batch: dl.mfcc_collate(
            batch,
            SR,
            MAX_LENGTH,
            TASK_IN,
            sub_task,
            pos_norm="zscore",
            hop_length=int(best_result.config["hopLen"]),
            n_mfcc=int(best_result.config["n_mfcc"]),
            n_fft=512,
        ),
    )

    intra_testloader = dl.testLoader(
        intra_TestDataset,
        batch_size=BATCH_SIZE,
        collate_fn=lambda batch: dl.mfcc_collate(
            batch,
            SR,
            MAX_LENGTH,
            TASK_IN,
            sub_task,
            pos_norm="zscore",
            hop_length=int(best_result.config["hopLen"]),
            n_mfcc=int(best_result.config["n_mfcc"]),
            n_fft=512,
        ),
    )
    test_network = ResNet.simpleResNet(num_classes=num_classes)
    test_network.to(DEVICE)
    best_chkpt = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

    model_state, _ = torch.load(best_chkpt)
    test_network.load_state_dict(model_state)

    with torch.no_grad():
        inter_score, inter_truth, inter_pred = customTest.test_model(
            DEVICE, main_task, inter_testloader, test_network
        )
        intra_score, intra_truth, intra_pred = customTest.test_model(
            DEVICE, main_task, intra_testloader, test_network
        )
    testLog = (
        f"Model: {MODEL_NAME}, Task: {TASK_IN}, inter score: {inter_score:>0.3}, "
        f"intra score: {intra_score:>0.3}, best trial config: {best_result.config}"
    )
    tuneRecord.info(testLog)
