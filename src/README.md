# Source files
The files within this directory are used to implement core functionalities for the various scripts one directory above.
# Preprocessing
Flow: data.dataset -> data.dataloaders + utils.signalprocessing -> train/test

## Experiment with dedicated preprocessing function
An experiment was conducted to identify if having the preprocessing function done as a collate function will be faster than integrating it within the dataset/dataloader.

A preliminary run showed that the integrated dataloader took 26.4s, while the dataloader without the preprocessing took 3.6s. When it comes to training, the training loop using the integrated loader took 1088.6s, while that for the dedicated loader with preprocessing as collate function took 1103.8s. The total time calculated for both are as shown below:\
Integrated dataloader with preprocessing: 1115.0s.\
Dataloader with preprocessing as collate function: 1106.6s (0.7% faster overall).