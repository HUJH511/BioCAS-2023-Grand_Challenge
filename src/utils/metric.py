from sklearn.metrics import confusion_matrix

def calc_score(truth, preds, verbose, task):
    """Get score of during training.
    Args:
        truth: Actual/correct labels.
        preds: Output obtained from model.
        verbose: True to enable verbosity. 
    Returns:
        score: Training score.
        SE: Sensitivity.
        SP: Specificity
        AS: Average Score
        HS: Harmonic Score
    """
    cm = confusion_matrix(y_true=truth, y_pred=preds)

    SP = cm.diagonal()[0] / cm[0].sum()
    SE = cm.diagonal()[task:].sum() / cm[task:,:].sum()
    AS = (SP + SE) / 2
    HS = (2 * SP * SE) / (SE + SP)
    score = (AS + HS) / 2

    if verbose:
        print("Sensitivity (SE): {:0.4f}".format(SE))
        print("Specificity (SP): {:0.4f}".format(SP))
        print("Average Score (AS): {:0.4f}".format(AS))
        print("Harmonic Score (HS): {:0.4f}".format(HS))
        print("Score: {:0.4f}".format(score))

    return score, SE, SP, AS, HS