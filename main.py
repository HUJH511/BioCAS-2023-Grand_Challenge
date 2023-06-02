# Brief flow for main
# Reference: https://github.com/chenzizhao/biocas-challenge-22 - Over-the-top

import os
import torch
import json
import argparse
import pathlib
import torchaudio
import numpy as np
import src.models as modelzoo
import src.data.dataloader as dl
from src.data.dataloader import preprocessing
from os import listdir

MODELS_PATH = {
    "Task_11": "models/11",
    "Task_12": "models/12",
    "Task_21": "models/21",
    "Task_22": "models/22",
}


def get_model(config, device):
    model_name = config["Name"]
    num_classes = config["Classes"]
    if "CNN" in model_name:
        classifier = modelzoo.leanCNN( 
            input_fdim=config["fdim"], 
            input_tdim=config["tdim"],
            num_class=num_classes,
        ).to(device)
    elif "ResNet" in model_name:
        classifier = modelzoo.PRS_Model(
            model="resnet18",
            head=config["head"],
            embedding_size=config["feat_dim"],
            num_classes=num_classes,
        ).to(device)
    else:
        print("Error in processing")
    return classifier


def write_json(content, json_path):
    with open(json_path, "w") as output_file:
        json.dump(content, output_file, indent=4, sort_keys=False)
    return


def main(args):
    task_in = "Task_" + args.task
    data_path = args.wav
    output_path = args.out

    dir_path = MODELS_PATH[task_in]
    model_checkpoint = os.path.join(dir_path, "model.pt")
    checkpoint_loader = torch.load(model_checkpoint)
    model_state = checkpoint_loader["model_state_dict"]

    config_file = os.path.join(dir_path, "config.json")
    with open(config_file, "r") as j:
        config = json.loads(j.read())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = get_model(config, device)
    model.load_state_dict(model_state)
    model.eval()

    output_log = {}
    with torch.no_grad():
        for filename in listdir(data_path):
            audio, sr = torchaudio.load(os.path.join(data_path, filename))
            audio = np.reshape(audio.numpy(), (1, -1))
            _, processed_audio = preprocessing(
                audio,
                sr,
                config["max_length"],
                config["feature"],
                **config["parameters"][0]
            )
            # split mfcc for task 2
            processed_audio = torch.unsqueeze(processed_audio, 0)
            raw_out = model(processed_audio.to(device))
            classification = torch.argmax(raw_out).item()
            classification = dl.idx2label(classification, task_in)
            output_log[filename] = classification
            
    write_json(output_log, output_path)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main for Evaluation")
    parser.add_argument(
        "--task",
        type=str,
        choices=["11", "12", "21", "22"],
        required=True,
    )
    parser.add_argument("--wav", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    main(args)
