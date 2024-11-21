import sys
sys.path.append('C:/Users/user/Text2LIVE/Prefix_AAC_ICASSP2023')
from AAC_util import *
from AAC_Prefix.AAC_Prefix import *  # network
from Train import *


import random
from argparse import ArgumentParser
import datetime
from pathlib import Path

import imageio
import numpy as np
import torch
import yaml
from tqdm import tqdm
import torchaudio

from datasets.image_dataset import SingleImageDataset
from models.clip_extractor import ClipExtractor
from models.image_model import Model
from util.losses import LossG
from util.util import tensor2im, get_optimizer

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if USE_CUDA else 'cpu')


def train_model(config): #모델 학습 함수

    # set seed
    seed = config["seed"] 
    if seed == -1:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    print(f"running with seed: {seed}.")

    # create dataset, loader
    dataset = SingleImageDataset(config) 

    # define model
    model = Model(config)

    # define loss function
    clip_extractor = ClipExtractor(config)
    criterion = LossG(config, clip_extractor)

    # define optimizer, scheduler
    optimizer = get_optimizer(config, model.parameters())

    for epoch in tqdm(range(1, config["n_epochs"] + 1)):
        inputs = dataset[0]
        for key in inputs:
            if key != "step":
                inputs[key] = inputs[key].to(config["device"])
        optimizer.zero_grad()
        outputs = model(inputs)
        for key in inputs:
            if key != "step":
                inputs[key] = [inputs[key][0]]
        losses = criterion(outputs, inputs)
        loss_G = losses["loss"]
        log_data = losses
        log_data["epoch"] = epoch

        # log current generated image to wandb
        if epoch % config["log_images_freq"] == 0:
            src_img = dataset.get_img().to(config["device"])
            with torch.no_grad():
                output = model.render(model.netG(src_img), bg_image=src_img)
            for layer_name, layer_img in output.items():
                image_numpy_output = tensor2im(layer_img)
                log_data[layer_name] = [wandb.Image(image_numpy_output)] if config["use_wandb"] else image_numpy_output

        loss_G.backward()
        optimizer.step()

        # update learning rate
        if config["scheduler_policy"] == "exponential":
            optimizer.param_groups[0]["lr"] = max(config["min_lr"], config["gamma"] * optimizer.param_groups[0]["lr"])
        lr = optimizer.param_groups[0]["lr"]
        log_data["lr"] = lr

        if config["use_wandb"]:
            wandb.log(log_data)
        else:
            if epoch % config["log_images_freq"] == 0:
                save_locally(config["results_folder"], log_data)

def save_locally(results_folder, log_data):
    path = Path(results_folder, str(log_data["epoch"]))
    path.mkdir(parents=True, exist_ok=True)
    for key in log_data.keys():
        if key in ["composite", "alpha", "edit_on_greenscreen", "edit"]:
            imageio.imwrite(f"{path}/{key}.png", log_data[key])

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        default="./configs/image_config.yaml",
        help="Config path",
    )
    parser.add_argument(
        "--example_config",
        default="golden_horse.yaml",
        help="Example config name",
    )

    parser.add_argument(
        "--audio_file",  # 오디오 파일 경로 추가
        required=True,
        help="Path to the audio file",
    )
    args = parser.parse_args()
    config_path = args.config

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(f"./configs/image_example_configs/{args.example_config}", "r") as f:
        example_config = yaml.safe_load(f)
    config["example_config"] = args.example_config
    config.update(example_config)

    audio_file_path = args.audio_file  # 명령줄에서 받은 오디오 파일 경로
    SAMPLE_RATE = 16000
    set_length = 30

    audio_file, _ = torchaudio.load(audio_file_path)  # 오디오 파일 로드

    if audio_file.shape[1] > (SAMPLE_RATE * set_length):
        audio_file = audio_file[:SAMPLE_RATE * set_length]
    # zero padding
    if audio_file.shape[1] < (SAMPLE_RATE * set_length):
        pad_len = (SAMPLE_RATE * set_length) - audio_file.shape[1]
        pad_val = torch.zeros((audio_file.shape[0], pad_len))
        audio_file = torch.cat((audio_file, pad_val), dim=1)

    # 오디오 파일 차원 조정
    if len(audio_file.size()) == 3:
        audio_file = audio_file.unsqueeze(0)

    audio_file = audio_file.to(device)

    model = get_model_in_table(1, 1, device)

    # 모델을 평가 모드로 설정
    model.eval()

    pred_caption = model(audio_file, None, beam_search=True)[0][0]  # 캡션 생성
    print("Caption :", pred_caption)

    config['comp_text'] = pred_caption  # This is where the text is injected
    config['screen_text'] = pred_caption

    run_name = f"-{config['image_path'].split('/')[-1]}"
    if config["use_wandb"]:
        import wandb

        wandb.init(project=config["wandb_project"], entity=config["wandb_entity"], config=config, name=run_name)
        wandb.run.name = str(wandb.run.id) + wandb.run.name
        config = dict(wandb.config)
    else:
        now = datetime.datetime.now()
        run_name = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}{run_name}"
        path = Path(f"{config['results_folder']}/{run_name}")
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "config.yaml", "w") as f:
            yaml.dump(config, f)
        config["results_folder"] = str(path)

    train_model(config)
    if config["use_wandb"]:
        wandb.finish()
