import learning.trainer as lt
import pytorch_lightning as pl
import os
from torch.multiprocessing import set_start_method

if __name__ == "__main__":
    trainer, opt = lt.load_trainer(load=True)

    trainer.test(
        opt["model"],
        opt["dataloader_test"],
        ckpt_path=opt["best_checkpoint_filename"],
    )
