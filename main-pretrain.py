from __future__ import print_function

import os
import time
import math
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

from src.train import train_supcon, valid_supcon
from src.utils.supcontrast import TwoCropTransform, SupConLoss
from src.utils.supcontrast import adjust_learning_rate, set_optimizer, save_model

import src.models as mdl
import src.data.dataset as ds
import src.data.dataloader as dl


# -------------------- Globals --------------------#
# Device Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {DEVICE} on {torch.cuda.get_device_name(0)} :D ")

# Dataset Config
MAX_LENGTH = 3.0
SR = 8000
HOP_LENGTH = 128
MAX_LENGTH_SAMPLES = int(MAX_LENGTH * SR / HOP_LENGTH)
INPUT_X_DIM = int(MAX_LENGTH * SR / HOP_LENGTH)
N_F_BIN = 64
N_FFT = 512
FEATURE = "mfcc"

# Log Config
formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s:%(message)s")


# -------------------- User-defined functions --------------------#
def setupLogger(name, logPath, level=logging.INFO):
    handler = logging.FileHandler(logPath)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


# -------------------- Set_Args() functions --------------------#
def set_args(opt):
    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epoch)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    # set the path according to the environment
    task_in = 'Task_' + opt.task
    opt.model_path = "./ckpts/PreTrain-Models/{}".format(task_in)
    opt.model_name = "{}_{}{}_{}{}_hop{}_{}_lr{}_temp{}_drop{}_val{}".format(
        opt.model, 
        FEATURE, 
        N_F_BIN, 
        opt.head,
        opt.embedding_size, 
        HOP_LENGTH,
        opt.optimizer,
        opt.learning_rate,
        opt.temperature,
        opt.dropout,
        opt.val_percent,
    )
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


# -------------------- Set_Loader() function definition --------------------#
def set_loader(opt):
    task_in = 'Task_' + opt.task
    data_path = opt.data_path

    data_dict={
        "train":[
            os.path.join(data_path, "train_wav"), 
            os.path.join(data_path, "train_json")
        ],
        "intra_test":[
            os.path.join(data_path, "test_wav"), 
            os.path.join(data_path, "test_json/intra_test_json")
        ],
        "inter_test":[
            os.path.join(data_path, "test_wav"),
            os.path.join(data_path, "test_json/inter_test_json")
        ],
    }

    main_task = int(task_in[-2])
    sub_task = int(task_in[-1])
    trainDataset, *_ = ds.genDatasets(
        task=main_task, 
        data_dict=data_dict,
        resample=None,
        feature=FEATURE,
        pre_emph=False,
        pos_norm="zscore",
        n_mfcc=N_F_BIN,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
    )
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(
            size=(N_F_BIN, MAX_LENGTH_SAMPLES), 
            padding=0, pad_if_needed=True
        ),
    ])

    print("\n\nGenerating Dataloader for SupCon Train Dataset...")
    supcon_loader = dl.trainValLoader(
        trainDataset,
        sub_task,
        valid_size=opt.val_percent,
        batch_size=opt.batch_size,
        collate_fn=lambda batch: dl.supcon_collate(
            batch, task_in, sub_task, transform=TwoCropTransform(train_transform)
        ),
        train_sampler="balanced",
        val_sampler="balanced",
    )
    return supcon_loader


# -------------------- Set_Model() function definition --------------------#
def set_model(opt):
    model = mdl.SupConResNet(
        name=opt.model, 
        head=opt.head,
        feat_dim=opt.embedding_size,
        dropout=opt.dropout,
    )
    criterion = SupConLoss(temperature=opt.temperature)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
    
    return model, criterion


# -------------------- Main() function definition --------------------#
def main(opt):
    # setup logger
    log_path = opt.log_path
    if not os.path.exists(log_path):
        open(log_path, "a").close()
    logger = setupLogger("ResultsLogger", log_path)

    # build data loader
    supcon_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    print("\n\nTraining...")
    print("Running for {} epochs...".format(opt.epoch))
    best_loss = 0
    best_epoch = 1
    # training routine
    for epoch in range(1, opt.epoch + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        train_loss = train_supcon(supcon_loader["train"], model, criterion, optimizer, epoch, opt)
        valid_loss = valid_supcon(supcon_loader["val"], model, criterion, opt)
        time2 = time.time()
        print("epoch {}, total time {:.2f}, train loss: {:.2f}, valid loss: {:.2f}".format(epoch, time2 - time1, train_loss, 1/valid_loss))

        if valid_loss > best_loss:
            best_loss = valid_loss
            best_model = model
            best_optimizer = optimizer
            best_epoch = epoch

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, "ckpt_epoch_{epoch}.pth".format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the best model
    save_file = os.path.join(opt.save_folder, "best.pth")
    save_model(best_model, best_optimizer, opt, opt.epoch, save_file)

    # log results
    log_msg = (
        f"Training - Model: {opt.model}_{FEATURE}{N_F_BIN}, Task: {opt.task}. Epoch: {opt.epoch}, "
        f"Last train loss: {train_loss:>0.2f}, "
        f"Best valid loss: {1/best_loss:>0.2f} at epoch {best_epoch}, "
        f"Method: {opt.method}. Temperature: {opt.temperature:>0.2f}, "
        f"Embedding size: {opt.head}{opt.embedding_size}, "
        f"Hop length: {HOP_LENGTH}, Optimizer: {opt.optimizer}, "
        f"Learning rate: {opt.learning_rate}, Dropout: {opt.dropout} "
    )
    logger.info(log_msg)
    return 


# -------------------- Script Run --------------------#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #   Add parser arguments below >>>>>

    # Dataset Config
    parser.add_argument(
        "--task",
        type=str,
        default="11",
        help="Task for script run - 11(default).",
    )
    parser.add_argument(
        "--data_path", type=str, default="SPRSound/", help="Directory for data."
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="logs/pretrain.logs",
        help="Path of file to save logs.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch_size"
    )
    parser.add_argument(
        "--val_percent", type=float, default=0.2, help="val_percent"
    )

    # Model Config
    parser.add_argument(
        "--model", type=str, default="resnet18", help="Name of the model chosen."
    )
    parser.add_argument(
        "--embedding_size", type=int, default=128, help="Embedding size"
    )
    parser.add_argument(
        "--head", type=str, default="linear", help="Embedding method"
    )

    # Train Config
    parser.add_argument(
        "--print_freq", type=int, default=50, help="print frequency"
    )
    parser.add_argument(
        "--save_freq", type=int, default=50, help="save frequency"
    )
    parser.add_argument(
        "--epoch", type=int, default=100, help="Epoch number"
    )

    # Optim Config
    parser.add_argument(
        "--optimizer", type=str, default="SGD", help="optimizer"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument(
        "--lr_decay_epochs", 
        type=str, 
        default="70,80,90", 
        help="where to decay lr, can be a list",
    )
    parser.add_argument(
        "--lr_decay_rate", 
        type=float, 
        default=0.1,
        help="decay rate for learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="weight decay"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="momentum"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.25, help="dropout"
    )

    # SupCon Config
    parser.add_argument(
        "--method", 
        type=str, 
        default="SupCon",
        choices=["SupCon", "SimCLR"], 
        help="choose method"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.1, 
        help="temperature for loss function"
    )

    # Other Config
    parser.add_argument(
        "--cosine", 
        action=argparse.BooleanOptionalAction, 
        help="using cosine annealing"
    )
    parser.add_argument(
        "--warm", 
        action=argparse.BooleanOptionalAction, 
        help="warm-up for large batch training"
    )
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)

    #   End of parser arguments <<<<<
    args = set_args(parser.parse_args())
    print(args)
    main(args) 