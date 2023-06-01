import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from src.utils.metric import calc_score


def test_supcon(SupCon, classifier, testloader, opt):
    targets = []
    embeddings = torch.zeros((0, opt.embedding_size), dtype=torch.float32)
    for data, label, _ in testloader:
        if torch.cuda.is_available():
            data = data.cuda(non_blocking=True)
        embedding = SupCon(data)
        targets.extend(label.detach().cpu().tolist())
        embeddings = torch.cat((embeddings, embedding.detach().cpu()), dim=0)

    x_embed = np.array(embeddings)
    y = np.array(targets)

    predictions = classifier.predict(x_embed)
    test_score, *_ = calc_score(y, predictions, verbose=True, task=int(opt.task_in[-2]))
    return test_score


def test_model(device, task, dataloader, trained_model, verbose=False, spike=False):
    """Post Evaluation Metric Platfrom. Feed in the trained model
    and train/validation data loader.
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
    trained_model.eval()
    for data, label, info in dataloader:
        data, label, info = data.to(device), label.to(device), info.to(device)
        outputs = trained_model(data)
        if spike:
            _, idx = outputs[0].sum(dim=0).max(1)
            preds.append(idx.cpu().numpy().tolist())
        else:
            _, predicted = torch.max(outputs, 1)
            preds.append(predicted.cpu().numpy().tolist())
        truth.append(label.cpu().numpy().tolist())

    preds_flat = [item for sublist in preds for item in sublist]
    truth_flat = [item for sublist in truth for item in sublist]

    if verbose == True:
        print("\nEvaluating....")
        print("Accuracy:", accuracy_score(truth_flat, preds_flat))
        print(classification_report(truth_flat, preds_flat))

    score, SE, SP, AS, HS = calc_score(
        truth_flat, preds_flat, verbose, task=task
    )

    # return accuracy_score(truth_flat, preds_flat), truth_flat, preds_flat
    return score, truth_flat, preds_flat
