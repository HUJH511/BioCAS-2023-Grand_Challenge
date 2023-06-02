from __future__ import print_function

import os
import logging
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import snntorch.functional as SF

import src.models as mdl
import src.data.dataset as ds
import src.data.dataloader as dl

from src.train import train_model
from src.test import test_model


# -------------------- Globals --------------------#
# Device Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {DEVICE} on {torch.cuda.get_device_name(0)} :D ")

# Feature Config
MAX_LENGTH = 3.0
SR = 8000
HOP_LENGTH = 128
MAX_LENGTH_SAMPLES = int(MAX_LENGTH * SR / HOP_LENGTH)
INPUT_X_DIM = int(MAX_LENGTH * SR / HOP_LENGTH)
N_F_BIN = 64
N_FFT = 512
FEATURE = "mfcc"

# Train Config
STRATEGY = ["score", "accuracy", "loss"]

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
    # set the path according to the environment
    # NEED MANUALLY CONFIG
    opt.ckpt = "best.pth"
    task_in = "Task_" + opt.task
    task_path = "./ckpts/PreTrain-Models/{}".format(task_in)
    model_name = "resnet18_{}{}_linear512_hop{}_SGD_lr0.001_temp0.1_drop0.25_val{}".format(
        FEATURE, 
        N_F_BIN, 
        HOP_LENGTH,
        opt.val_percent,
    )
    opt.model_path = os.path.join(task_path, model_name)
    if not os.path.isdir(opt.model_path):
        raise Exception("Wrong PreTrain Model")

    return opt


# -------------------- Set_Loader() function definition --------------------#
def set_loader(opt):
    task_in = "Task_" + opt.task
    main_task = int(task_in[-2])
    sub_task = int(task_in[-1])
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

    trainDataset, intra_testDataset, inter_testDataset = ds.genDatasets(
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

    print("\n\nGenerating Dataloader for Train Dataset...")
    dataloader = dl.trainValLoader(
        trainDataset,
        sub_task,
        valid_size=opt.val_percent,
        batch_size=opt.batch_size,
        collate_fn=lambda batch: dl.custom_collate(
            batch, MAX_LENGTH_SAMPLES, task_in, sub_task
        ),
        train_sampler="balanced",
        val_sampler="balanced",
    )

    print("\n\nGenerating Dataloader for Intra Dataset...")
    intra_testloader = dl.testLoader(
        intra_testDataset,
        batch_size=opt.batch_size,
        collate_fn=lambda batch: dl.custom_collate(
            batch, MAX_LENGTH_SAMPLES, task_in, sub_task
        ),
        shuffle_in=False,
    )

    print("\nGenerating Dataloader for Inter Dataset...")
    inter_testloader = dl.testLoader(
        inter_testDataset,
        batch_size=opt.batch_size,
        collate_fn=lambda batch: dl.custom_collate(
            batch, MAX_LENGTH_SAMPLES, task_in, sub_task
        ),
        shuffle_in=False,
    )

    return dataloader, intra_testloader, inter_testloader


# -------------------- Set_Model() function definition --------------------#
def get_model(opt, model, num_classes):
    if model == "SupCon":
        PRS_model = mdl.PRS_classifier(opt, num_classes=num_classes, pretrain=True).to(DEVICE)

        for param in PRS_model.encoder.parameters():
            param.requires_grad = False

        for param in PRS_model.classifier.parameters():
            param.requires_grad = True

        loss_fn = nn.NLLLoss()
        spike = False
        return PRS_model, loss_fn, spike
    
    elif model == "ResNet":
        PRS_classifier = mdl.simpleResNet(
            num_classes=num_classes,
        ).to(DEVICE)
        loss_fn = nn.NLLLoss()
        spike = False
        return PRS_classifier, loss_fn, spike
    
    elif model == "CNN":
        PRS_classifier = mdl.leanCNN(
            num_class=num_classes,
            input_fdim=N_F_BIN,
            input_tdim=INPUT_X_DIM,
        ).to(DEVICE)
        loss_fn = nn.NLLLoss()
        spike = False
        return PRS_classifier, loss_fn, spike
    
    elif model == "SNN":
        PRS_classifier = mdl.customSNet(
            num_steps=10, beta=0.5, num_class=num_classes
        ).to(DEVICE)
        loss_fn = SF.loss.ce_count_loss(False, False)
        spike = True
        return PRS_classifier, loss_fn, spike
    
    else:
        raise Exception(
            "No such model, please select within (SupCon, ResNet, CNN, SNN)"
            )
    

# -------------------- Main() function definition --------------------#
def main(args):
    log_path = args.log_path
    mode = args.mode
    if not os.path.exists(log_path):
        open(log_path, "a").close()
    logger = setupLogger("ResultsLogger", log_path)
    task_in = "Task_" + args.task
    main_task = int(task_in[-2])

    dataloader, intra_testloader, inter_testloader = set_loader(args)

    num_classes = len(dl.classes[task_in])

    if mode == "train":
        ### Start of model and task customization >>>>>
        lr = args.learning_rate
        epoch_num = args.epoch
        model_name = args.model
        PRS_classifier, loss_fn, spike = get_model(args, model_name, num_classes)
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, PRS_classifier.parameters()), 
            lr=lr, 
            weight_decay=1e-4
        )
        best_dict = train_model(
            device=DEVICE,
            task=task_in,
            dataloader=dataloader,
            model=PRS_classifier,
            criterion=loss_fn,
            optimizer=optimizer,
            n_epochs=epoch_num,
            print_every=1,
            verbose=True,
            plot_results=False,
            validation=True,
            save_ckpt=False,
            spike=spike,
            model_name=model_name,
            strategy=STRATEGY,
        )
        ### End of model and task customization <<<<<
        ### Start of evaluation for testdatasets >>>>>
        for item in STRATEGY:
            test_classifier, _, spike = get_model(args, model_name, num_classes)
            test_classifier.load_state_dict(best_dict[item]["model_state_dict"])
            test_classifier.eval()

            inter_score, *_ = test_model(
                device=DEVICE,
                task=main_task,
                dataloader=inter_testloader,
                trained_model=test_classifier,
            )
            intra_score, *_ = test_model(
                device=DEVICE,
                task=main_task,
                dataloader=intra_testloader,
                trained_model=test_classifier,
            )
            test_score = (inter_score + intra_score) / 2

            epoch, _, _, score, acc, loss = best_dict[item].values()
            valloss = 1 / loss
            log_msg = (
                f"Training - Model: {model_name}_{FEATURE}, Task: {task_in}. Epoch: {epoch_num}, "
                f"Strategy: {item}, "
                f"TestScore: {test_score:>0.2f} at epoch {epoch}, "
                f"ValScore: {score:>0.2f}, "
                f"ValAccuracy: {acc:>0.2f}, "
                f"ValLoss: {valloss:>0.2f}, "
                f"IntraScore: {intra_score:>0.2f}, "
                f"InterScore: {inter_score:>0.2f} "
            )
            logger.info(log_msg)
        ### End of evaluation and logging results <<<<<
        if args.save_model:
            PATH = "ckpts/FineTune-Models/{}_{}.pt".format(model_name, task_in)
            torch.save(best_dict, PATH)
            
    else:
        print("\nTesting...")
        model_name = args.model
        strategy = args.strategy

        PRS_classifier, _, spike = get_model(args, model_name, num_classes)
        PATH = "ckpts/FineTune-Models/{}_{}.pt".format(model_name, task_in)
        CheckPoint = torch.load(PATH)
        PRS_classifier.load_state_dict(CheckPoint[strategy]["model_state_dict"])
        PRS_classifier.eval()

        print("\nResults for Intra")
        _ = test_model(
            device=DEVICE,
            task=main_task,
            dataloader=intra_testloader,
            trained_model=PRS_classifier,
            verbose=True,
            spike=spike,
        )
        print("\nResults for Inter")
        _ = test_model(
            device=DEVICE,
            task=main_task,
            dataloader=inter_testloader,
            trained_model=PRS_classifier,
            verbose=True,
            spike=spike,
        )
        
        txt = input("Enter Y/N to get final model: ")
        if txt == "Y" or "y":
            FINAL_PATH = os.path.join("models", args.task, "model.pt")
            torch.save(best_dict["score"], FINAL_PATH)
    return


# -------------------- Script Run --------------------#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #   Add parser arguments below >>>>>
 
    # Mode Config
    parser.add_argument("--mode", type=str, default="train", help="Train or test mode.")

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
        default="logs/finetune.logs",
        help="Path of file to save logs.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for dataloaders."
    )
    parser.add_argument(
        "--val_percent",
        type=float,
        default=0.2,
        help="Fraction of dataset for validation (0-1)",
    )
    
    # Model Config
    parser.add_argument(
        "--model", type=str, default="SupCon", help="Type of the model chosen."
    )
    parser.add_argument("--save_model", action=argparse.BooleanOptionalAction)

    # Train Config
    parser.add_argument(
        "--epoch", type=int, default=20, help="Epoch number."
    )

    # Optim Config
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="score",
        help="Choose strategy to do evaluation.",
    )
    
    #   End of parser arguments <<<<<
    args = set_args(parser.parse_args())
    print(args)
    main(args)