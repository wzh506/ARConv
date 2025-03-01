import os
import argparse
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import scipy.io as sio
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.data import DataSet

from .models import ARNet

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True

def save_checkpoint(
    model, optimizer, scheduler, epoch, save_path
):  # save model function
    check_point = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
    }
    save_path = (
        save_path
        + "/"
        + f"checkpoint_{epoch}_"
        + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        + ".pth"
    )
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(check_point, save_path)


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs, lr, ckpt, batch_size, hw_range, task, checkpoint_save_path = (
        config.epochs, config.lr, config.ckpt, config.batch_size,
        config.hw_range, config.task, config.checkpoint_save_path
    )
    train_set_path, checkpoint_path = config.train_set_path, config.checkpoint_path
    train_set = DataSet(file_path=train_set_path)
    training_data_loader = DataLoader(
        dataset=train_set,
        num_workers=0,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
    )
    if task == "wv3":
        pan_channels, lms_channels = 1, 8
    elif task in ["qb", "gf2"]:
        pan_channels, lms_channels = 1, 4

    model = ARNet(pan_channels, lms_channels).to(device)

    model = nn.DataParallel(model)
    criterion = nn.L1Loss().to(device)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.999)
    )
    scheduler = StepLR(optimizer, step_size=200, gamma=0.8)
    epoch = 1

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]
        scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"=> successfully loaded checkpoint from '{checkpoint_path}'")

    print("Start training...")
    while epoch <= epochs + 1:
        epoch_train_loss = []
        model.train()
        pbar = tqdm(
            enumerate(training_data_loader),
            total=len(training_data_loader),
            bar_format="{l_bar}{bar:10}{r_bar}",
        )
        for iteration, batch in pbar:
            gt = batch[0].to(device)
            lms = batch[1].to(device)
            pan = batch[4].to(device)
            optimizer.zero_grad()
            output = model(pan, lms, epoch, hw_range=hw_range)
            loss = criterion(output, gt)
            epoch_train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        t_loss = np.nanmean(np.array(epoch_train_loss))
        print("Epoch: {}/{} training loss: {:.7f}".format(epochs, epoch, t_loss))
        scheduler.step()
        if epoch % ckpt == 0 or (epoch - 1) % 100 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_save_path)
        print("lr: ", optimizer.param_groups[0]["lr"])
        f = open("loss.txt", "a")
        f.write(f"epoch: {epoch} | train_loss: {t_loss}\n")
        epoch = epoch + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size used in the training and validation loop.",
    )
    parser.add_argument(
        "--epochs", default=600, type=int, help="Total number of epochs."
    )
    parser.add_argument(
        "--lr",
        default=0.0006,
        type=float,
        help="Base learning rate at the start of the training.",
    )
    parser.add_argument(
        "--ckpt", default=300, type=int, help="Save model every ckpt epochs."
    )
    parser.add_argument(
        "--train_set_path", default="", type=str, help="Path to the training set."
    )
    parser.add_argument(
        "--checkpoint_path", default="", type=str, help="Path to the checkpoint file."
    )
    parser.add_argument(
        "--checkpoint_save_path",
        default="",
        type=str,
        help="Path to the checkpoint file.",
    )
    parser.add_argument(
        "--hw_range",
        nargs=2,
        type=int,
        default=[0, 18],
        help="The range of the height and width.",
    )
    parser.add_argument("--use_pretrain", action="store_true", help="...")
    parser.add_argument(
        "--task",
        default="wv3",
        type=str,
        choices=["wv3", "qb", "gf2"],
        help="Model to train (choices: wv3, qb, gf2).",
    )
    config = parser.parse_args()
    train(config)
