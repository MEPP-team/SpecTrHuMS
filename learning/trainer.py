import argparse
import sys
import os
import numpy as np
import torch
import json
import pytorch_lightning as pl
import mmap
import shutil
import learning.custom_dataloader as custom_dataloader

from os import path
from utils.init_weights import init_weights
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from torch.utils.data import DataLoader


def get_opt(load):
    if not path.isdir("checkpoints/"):
        os.mkdir("checkpoints/")

    parser = argparse.ArgumentParser()

    if load:
        parser.add_argument("--load_job_id", required=True)
    else:
        parser.add_argument("--job_id", required=True)

    args, leftovers = parser.parse_known_args()

    if load:
        print("Loading options from " + "checkpoints/" +
              str(args.load_job_id) + "/infos.json")

        with open("checkpoints/" + str(args.load_job_id) + "/infos.json", "r") as outfile:
            opt = json.load(outfile)
    else:
        print("Loading default options for new job")

        with open("learning/default_options_trainer.json", "r") as outfile:
            opt = json.load(outfile)

        with open("learning/modules/default_options.json", "r") as outfile:
            opt.update(json.load(outfile))

    # so we can pass other default options as program argument
    for key, value in opt.items():
        parser.add_argument("--" + key, default=value, type=type(value))

    opt.update(vars(args))

    if not load:
        os.mkdir("checkpoints/" + str(opt["job_id"]))

        with open("checkpoints/" + str(opt["job_id"]) + "/infos.json", "w") as outfile:
            json.dump(opt, outfile, sort_keys=True, indent=4)

    print("Options:\n", json.dumps(opt, sort_keys=True, indent=4), end="\n\n")

    return opt


def load_trainer(load, profiler="simple"):
    seed_everything(42, workers=True)
    print()

    opt = get_opt(load)

    if opt["loss_type"] == "MSE":
        opt["loss"] = torch.nn.MSELoss()
    else:
        print("Loss type not implemented.")
        exit()

    opt["nb_vertices"] = 6890

    with open(opt["path_dataset"] + "infos.json", "r") as outfile:
        opt_dataset = json.load(outfile)
        opt["nb_freqs"] = opt_dataset["nb_freqs"]
        opt["framerate"] = opt_dataset["framerate"]

    path_evecs = "data/evecs_4096.bin"

    with open(path_evecs, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)

        evecs = (
            torch.tensor(np.frombuffer(mm[:], dtype=np.float32)).view(
                6890, 4096).to(opt["device"])
        )

        opt["evecs"] = evecs[:, : opt["nb_freqs"]]

    path_faces = "data/faces.bin"

    with open(path_faces, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        opt["faces"] = np.frombuffer(mm[:], dtype=np.intc).reshape(-1, 3)

    opt["dataloader_train"] = DataLoader(
        custom_dataloader.CustomDataset(opt, train_test="train"),
        batch_size=opt["train_batch_size"],
        shuffle=True,
        num_workers=opt["num_workers"],
        # prefetch_factor=opt["num_workers"],
        # pin_memory=True,
        # persistent_workers=True,
        collate_fn=custom_dataloader.collate_fn,
    )

    opt["dataloader_test"] = DataLoader(
        custom_dataloader.CustomDataset(opt, train_test="test"),
        batch_size=opt["test_batch_size"],
        shuffle=False,
        num_workers=opt["num_workers"],
        # prefetch_factor=opt["num_workers"],
        # pin_memory=True,
        # persistent_workers=True,
        collate_fn=custom_dataloader.collate_fn,
    )

    # model
    exec("from learning.modules." +
         opt["model_type"] + " import " + opt["model_type"])
    opt["model"] = eval(opt["model_type"])(opt).to(opt["device"])
    opt["model"].apply(init_weights)

    nb_params = sum(p.numel()
                    for p in opt["model"].parameters() if p.requires_grad)

    print("Number of parameters:", str(nb_params), end="\n\n")

    # pytorch lightning
    logger = TensorBoardLogger(
        save_dir="checkpoints",
        name="",
        version=str(opt["job_id"]),
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/" + str(opt["job_id"]),
        filename="{epoch:02d}_{MPJPE:.2f}",
        every_n_epochs=opt["check_val_every_n_epoch"],
        save_top_k=0,
        save_last=True,
        monitor="MPJPE",
        mode="min",
    )

    # if output is redirected to log file, print less
    if os.isatty(sys.stdout.fileno()):
        refresh_rate = 1
    else:
        refresh_rate = 1000

    progress_bar = TQDMProgressBar(
        refresh_rate=refresh_rate,
    )

    accelerator = "gpu" if opt["device"] == "cuda" else "cpu"

    print("Creating trainer: ")
    pl_trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        profiler=profiler,
        max_epochs=opt["num_iterations"],
        check_val_every_n_epoch=opt["check_val_every_n_epoch"],
        logger=logger,
        precision=32,
        default_root_dir="checkpoints/",
        callbacks=[checkpoint_callback, progress_bar],
        deterministic=True,
        benchmark=False,
    )

    print()

    if load:
        opt["best_checkpoint_filename"] = "checkpoints/" + \
            str(opt["load_job_id"]) + "/last.ckpt"

        print("Loading checkpoint:", opt["best_checkpoint_filename"])

        opt["model"] = (
            opt["model"]
            .load_from_checkpoint(opt["best_checkpoint_filename"], opt=opt)
            .to(opt["device"])
        )

    return pl_trainer, opt
