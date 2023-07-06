import learning.trainer as lt
from torch.multiprocessing import set_start_method

if __name__ == "__main__":
    trainer, opt = lt.load_trainer(load=False)

    trainer.fit(opt["model"], opt["dataloader_train"], opt["dataloader_test"])
