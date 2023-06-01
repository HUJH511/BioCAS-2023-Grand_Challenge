import numpy as np
import itertools
import matplotlib.pyplot as plt 

SMALL_SIZE = 22
MEDIUM_SIZE = 24
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Comment if using Colab
# plt.rcParams["font.family"] = "Times New Roman"

def plot_confusion_matrix(
        cm, 
        classes,
        normalize=False,
        title='Confusion matrix',
        cmap=plt.cm.Blues
):
    """Print and plot the confusion matrix.
    Args:
        cm: confusion matrix array.
        classes: array of classes/labels.
        normalize: True to show percentage value. (default is True).
        title: title of confusion matrix.
    Returns:
        plot of cunfusion matrix.
    """
    print('Visualizing...')
    if normalize: 
        count = cm.sum(axis=1)[:, np.newaxis]
        nonzero_count = np.where(count!=0, count, count+1)
        cm = cm.astype('float') / nonzero_count
        print("Normalized confusion matrix\n")
    else:
        print('Confusion matrix, without normalization\n')
    
    plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize = 30)
    plt.colorbar(fraction=0.046)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes.values(), rotation=45)
    plt.yticks(tick_marks, classes.values())
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')