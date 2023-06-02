from __future__ import print_function

import sys
import time
import torch
import matplotlib.pyplot as plt
import snntorch.functional as SF
from sklearn.metrics import accuracy_score, classification_report

from src.utils.metric import calc_score
from src.utils.supcontrast import AverageMeter
from src.utils.supcontrast import warmup_learning_rate


def train_supcon(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt.method == "SupCon":
            loss = criterion(features, labels)
        elif opt.method == "SimCLR":
            loss = criterion(features)
        else:
            raise ValueError("contrastive method not supported: {}".
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if opt.verbose and (idx + 1) % opt.print_freq == 0:
            print("Train: [{0}][{1}/{2}]\t"
                  "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                  "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                  "loss {loss.val:.3f} ({loss.avg:.3f})".format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()
    return losses.avg


def valid_supcon(valid_loader, model, criterion, opt):
    """one epoch validating"""
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    with torch.no_grad():
        for idx, (images, labels, _) in enumerate(valid_loader):
            data_time.update(time.time() - end)

            images = torch.cat([images[0], images[1]], dim=0)
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            # compute loss
            features = model(images)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            if opt.method == "SupCon":
                loss = criterion(features, labels)
            elif opt.method == "SimCLR":
                loss = criterion(features)
            else:
                raise ValueError("contrastive method not supported: {}".
                                format(opt.method))
            # update metric
            losses.update(loss.item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    return 1/losses.avg


def train_model(
    device,
    task,
    dataloader,
    model,
    criterion,
    optimizer,
    spike=False,
    n_epochs=10,
    print_every=1,
    verbose=True,
    plot_results=True,
    validation=True,
    save_ckpt=False,
    model_name=None,
    strategy=["score"],
):
    """Basic training flow for neural network.
    Args:
        device: either cpu or cuda for acceleration.
        task: full task for score calculation.
        dataloader: data loader containing training data.
        model: network to train.
        criterion: loss function.
        optimizer: optimizer for weights update.
        spike: True to initiate spike-train process (default is False).
        n_epochs: number of epochs (default is 10).
        print_every: int for number of epoch before printing (default is 1).
        verbose: True to enable verbosity (default is True).
        plot_result: True to plot results (default is True).
        validation: True to run validation as well (default is True).
        save_model: True to save checkpoint to ckpts folder (default is False).
        model_name: the name of model to be trained (default is None).
    Returns:
        model: trained model.
        best_epoch: epoch which yield the best score.
        best_score: best score obtained.
        train_loss[-1]: the latest training loss.
        val_loss[-1]: the latest validation loss.
        best_info: detailed information which yield the best score.
    """
    best_dict = {}
    best_result = {}
    for item in strategy:
        best_result[item] = 0
    losses = []
    start = time.time()
    print("\nTraining for {} epochs...".format(n_epochs))
    for epoch in range(n_epochs):
        if verbose == True and epoch % print_every == 0:
            print("\n\nEpoch {}/{}:".format(epoch + 1, n_epochs))

        if validation == True:
            evaluation = ["train", "val"]
        else:
            evaluation = ["train"]

        # Each epoch has a training and validation phase
        for phase in evaluation:
            if phase == "train":
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            for data, label, info in dataloader[phase]:
                data, label, info = data.to(device), label.to(device), info.to(device)

                # forward + backward + optimize
                x = data
                outputs = model(x)  # spk_rec, mem_rec if spike=True
                acc = calc_accuracy(outputs, label, spike)
                loss = calc_loss(criterion, outputs, label, spike)  # loss function
                # zero the parameter (weight) gradients
                optimizer.zero_grad()

                # backward + optimize only if in training phase
                if phase == "train":
                    loss.backward()
                    # update the weights
                    optimizer.step()

                # record loss statistics
                running_loss += loss.item()

            losses.append(running_loss)

            if verbose == True and epoch % print_every == 0:
                print(
                    "{} loss: {:.4f} | acc: {:.4f}|".format(phase, running_loss, acc),
                    end=" ",
                )

        val_score, val_acc = valid_model(
            device=device,
            task=int(task[-2]),
            dataloader=dataloader[evaluation[-1]],
            trained_model=model,
            verbose=False,
            spike=spike,
        )

        val_results = {
            "score": val_score,
            "accuracy": val_acc,
            "loss": 1 / losses[-1],
        }

        for item in strategy:
            if val_results[item] > best_result[item]:
                best_result[item] = val_results[item]
                best_dict[item] = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                } | val_results

    if verbose == True:
        print("\nFinished Training  | Time:{}".format(time.time() - start))

    if plot_results == True:
        plt.figure(figsize=(10, 10))
        plt.plot(losses[0::2], label="train_loss")
        if validation == True:
            plt.plot(losses[1::2], label="validation_loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.draw()

    if save_ckpt:
        PATH = "ckpts/{}_CheckPoint_task{}.pt".format(model_name, task)
        torch.save(best_dict, PATH)

    return best_dict


def calc_accuracy(output, Y, spike=False):
    """Get batch accuracy of during training.
    Args:
        output: output obtained from model.
        Y: actual/correct labels.
        spike: True to use spiking output.
    Returns:
        train_acc: Training accuracy for that batch.
    """
    if spike:
        train_acc = SF.acc.accuracy_rate(output[0], Y)
    else:
        max_vals, max_indices = torch.max(output, 1)
        train_acc = (max_indices == Y).sum().data.cpu().numpy() / max_indices.size()[0]
    return train_acc


def calc_loss(criterion, output, Y, spike=False):
    """Get batch loss of during training.
    Args:
        criterion: loss function.
        output: output obtained from model.
        Y: actual/correct labels.
        spike: True to use spiking output.
    Returns:
        loss_fn(output, Y)
    """
    if spike:
        return criterion(output[0], Y)
    else:
        return criterion(output, Y)


def valid_model(device, task, dataloader, trained_model, verbose=False, spike=False):
    """Post Evaluation Metric Platfrom. Feed in the trained model and train/validation data loader.
    Args:
        device: either cpu or cuda for acceleration.
        dataloader: dataloader containing data for evaluation.
        trained_model: model used for evaluation.
        verbose: True to enable verbosity (True as default).
    Returns:
        classification accuracy obtained from sklearn's accuracy score.
    """
    truth = []
    preds = []
    for data, label, info in dataloader:
        data, label, info = data.to(device), label.to(device), info.to(device)
        x = data
        outputs = trained_model(x)
        if spike:
            _, idx = outputs[0].sum(dim=0).max(1)
            preds.append(idx.cpu().numpy().tolist())
        else:
            _, predicted = torch.max(outputs, 1)
            preds.append(predicted.cpu().numpy().tolist())
        truth.append(label.cpu().numpy().tolist())

    preds_flat = [item for sublist in preds for item in sublist]
    truth_flat = [item for sublist in truth for item in sublist]

    score, *_ = calc_score(truth_flat, preds_flat, verbose, task=task)
    accuracy = accuracy_score(truth_flat, preds_flat)
    if verbose == True:
        print("\nEvaluating....")
        print("Accuracy:", accuracy)
        print(classification_report(truth_flat, preds_flat))
    return score, accuracy
