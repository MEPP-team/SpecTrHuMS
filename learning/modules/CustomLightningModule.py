import warnings
import time
import pytorch_lightning as pl
import torch
import numpy as np
import utils.welford_means_stds as w
import mmap
import os

from utils.compute_edge_lengths import compute_edge_lengths

print("Ignoring warning concerning number of workers \n")

warnings.filterwarnings("ignore", ".*does not have many workers.*")


class CustomLightningModule(pl.LightningModule):
    def __init__(self, opt):
        super(CustomLightningModule, self).__init__()

        self.opt = opt

        # compute self.edge_indices
        edge_indices_0 = self.opt["faces"][:, 0][:, None]
        edge_indices_1 = self.opt["faces"][:, 1][:, None]
        edge_indices_2 = self.opt["faces"][:, 2][:, None]

        edge_indices_01 = np.concatenate(
            (edge_indices_0, edge_indices_1), axis=1)
        edge_indices_12 = np.concatenate(
            (edge_indices_1, edge_indices_2), axis=1)
        edge_indices_20 = np.concatenate(
            (edge_indices_2, edge_indices_0), axis=1)

        # edge_indices shape => (82656, 2)
        edge_indices = np.concatenate(
            (edge_indices_01, edge_indices_12, edge_indices_20), axis=0
        )

        # remove duplicates
        edge_indices = list(map(tuple, edge_indices))
        self.edge_indices = list({*map(tuple, map(sorted, edge_indices))})
        # self.edge_indices shape => (20664, 2)

        self.checkpoint_dir = "checkpoints/" + opt["job_id"] + "/"

        # for latent variables means and stds computation by batch
        self.welford = w.Welford(
            tuple([opt["size_latent"]]),
            self.checkpoint_dir + "means_stds.pt",
            opt["device"]
        )

        self.means_stds = self.welford.means_stds.clone()

        self.means_stds_loaded = False

        path_means_stds_job_id = self.checkpoint_dir + "means_stds.pt"

        print('trying to load ', path_means_stds_job_id)

        if os.path.isfile(path_means_stds_job_id):
            print("loading means and stds from ", path_means_stds_job_id)
            self.means_stds = torch.load(
                path_means_stds_job_id).to(opt["device"])
            self.means_stds_loaded = True
        else:
            print("not loading means and stds")

        # load SMPL models in order to compute corresponding skeletons
        bm_male = "data/smplh/male/model.npz"
        bm_female = "data/smplh/female/model.npz"
        bm_neutral = "data/smplh/neutral/model.npz"

        self.j_reg_male = (
            torch.tensor(np.load(bm_male)["J_regressor"]).float().to(
                opt["device"])
        )
        self.j_reg_female = (
            torch.tensor(np.load(bm_female)["J_regressor"]).float().to(
                opt["device"])
        )
        self.j_reg_neutral = (
            torch.tensor(np.load(bm_neutral)[
                         "J_regressor"]).float().to(opt["device"])
        )

        self.upsample = torch.nn.Upsample(scale_factor=2, mode="linear")

        self.all_mean_variance_input = []
        self.all_mean_variance_output = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.opt["learning_rate"],
            weight_decay=0,
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.opt["num_iterations"], eta_min=1e-6
        )

        scheduler = {
            "scheduler": lr_scheduler,
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def infer_static_dynamic(self, batch, train_test):
        # static module
        rand_frame = np.random.randint(0, self.opt["size_window"])
        inputs_static = batch[:, rand_frame, ...].clone()

        inputs_latent = self.enc_static(inputs_static)

        outputs_static = self.dec_static(inputs_latent)

        # if epoch 0, only return static and don't aggregate
        if self.trainer.current_epoch == 0 and train_test == "train":
            return (
                inputs_static,
                outputs_static,
                None,
                None
            )

        # if epoch 1, only return static and aggregate
        if self.trainer.current_epoch == 1 and train_test == "train":
            if train_test == "train":
                self.welford.aggregate(inputs_latent)

            return (
                inputs_static,
                outputs_static,
                None,
                None
            )

        # if epoch > 1, also infer dynamic module
        inputs_dynamic = batch.clone()

        with torch.no_grad():
            inputs_latent = self.enc_static(inputs_dynamic)

        if train_test == "train":
            self.welford.aggregate(inputs_latent)

        outputs_latent = self.enc_transformer(inputs_latent.detach())

        # standardize dynamic loss
        inputs_latent = inputs_latent - self.means_stds[0]
        inputs_latent = inputs_latent / self.means_stds[1]

        outputs_latent = outputs_latent - self.means_stds[0]
        outputs_latent = outputs_latent / self.means_stds[1]

        return (
            inputs_static,
            outputs_static,
            inputs_latent,
            outputs_latent
        )

    def on_train_epoch_start(self):
        self.welford.initialize_values()

        self.start_time = time.time()

    def training_step(self, train_batch, batch_idx):
        train_batch, _ = train_batch

        (
            inputs_static,
            outputs_static,
            inputs_latent,
            outputs_latent,
        ) = self.infer_static_dynamic(train_batch, "train")

        loss_static = self.opt["loss"](inputs_static, outputs_static)
        self.log("loss_static", loss_static)

        if self.trainer.current_epoch == 0 or self.trainer.current_epoch == 1:
            return {"loss": loss_static}

        loss_dynamic = self.opt["loss"](inputs_latent, outputs_latent)
        self.log("loss_dynamic", loss_dynamic)

        return {"loss": loss_static + loss_dynamic}

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.start_time
        self.log("epoch_time", epoch_time)
        self.log("learning_rate", self.lr_schedulers().get_last_lr()[0])

        self.welford.finalize()

        if torch.count_nonzero(self.welford.means_stds).cpu().item() != 0:
            self.means_stds = self.welford.means_stds.clone()

            if (self.trainer.current_epoch + 1) % self.opt[
                "check_val_every_n_epoch"
            ] == 0:
                self.welford.save()

    def on_validation_epoch_start(self):
        self.validation_test_start()

    def validation_step(self, val_batch, batch_idx):
        self.validation_test_step(val_batch)

    def on_validation_epoch_end(self):
        self.validation_test_end()

        self.log("MPJPE", np.mean(self.mpjpe[self.time_steps]))
        self.log("MELV", np.mean(self.melv_output[self.time_steps]))

    def on_test_start(self):
        self.validation_test_start()

    def test_step(self, test_batch, batch_idx):
        self.validation_test_step(test_batch)

    def on_test_end(self):
        self.validation_test_end()

        np.set_printoptions(precision=2)

        print("MPJPE each time step")
        print(self.mpjpe[self.time_steps])

        print()
        print("MELV input")
        print(self.melv_input[self.time_steps])

        print("MELV input std")
        print(
            torch.std(self.all_melv_input[:, self.time_steps], dim=0).cpu().numpy())

        print()
        print("MELV output")
        print(self.melv_output[self.time_steps])

        print("MELV output std")
        print(
            torch.std(self.all_melv_output[:, self.time_steps], dim=0).cpu().numpy())

    def validation_test_start(self):
        self.nb_samples = 0

        self.time_steps = [1, 3, 7, 9, 13, 17, 21, 24]

        self.mpjpe = np.zeros([25])  # joints
        self.melv_input = np.zeros([25])  # each time step MELV input
        self.melv_output = np.zeros([25])  # each time step MELV output

    def validation_test_step(self, batch):
        coeffs, genders = batch

        self.nb_samples += coeffs.shape[0]

        j_regs = []
        for i in genders:
            if i == 0:
                j_regs.append(self.j_reg_male)
            elif i == 1:
                j_regs.append(self.j_reg_female)
            elif i == 2:
                j_regs.append(self.j_reg_neutral)
            else:
                print("wrong gender")
                exit()

        j_regs = torch.stack(j_regs).unsqueeze(1)

        (
            _,
            _,
            _,
            outputs_latent,
            # outputs_dynamic,
        ) = self.infer_static_dynamic(coeffs, "test")

        # destandardize
        outputs_latent = outputs_latent * self.means_stds[1]
        outputs_latent = outputs_latent + self.means_stds[0]

        outputs_dynamic = self.dec_static(outputs_latent)

        del outputs_latent

        # take last known frame for edge computation
        coeffs = coeffs[:, 49:]
        outputs_dynamic = outputs_dynamic[:, 50:]

        vertices_input_dynamic = torch.matmul(self.opt["evecs"], coeffs)
        vertices_output_dynamic = torch.matmul(
            self.opt["evecs"], outputs_dynamic)

        del coeffs
        del outputs_dynamic

        edge_lengths_input = compute_edge_lengths(
            vertices_input_dynamic, self.edge_indices
        )

        edge_lengths_gt = edge_lengths_input[:, 0].clone().unsqueeze(1)

        vertices_input_dynamic = vertices_input_dynamic[:, 1:]
        edge_lengths_input = edge_lengths_input[:, 1:]

        edge_lengths_output = compute_edge_lengths(
            vertices_output_dynamic, self.edge_indices
        )

        # compute edge lengths difference
        self.compute_edge_infos(edge_lengths_input, edge_lengths_gt, "input")
        self.compute_edge_infos(edge_lengths_output, edge_lengths_gt, "output")

        # compute joints positions and remove hand joints
        joints_input = torch.matmul(j_regs, vertices_input_dynamic)[:, :, :22]
        joints_output = torch.matmul(
            j_regs, vertices_output_dynamic)[:, :, :22]

        del vertices_input_dynamic
        del vertices_output_dynamic

        mpjpe = torch.sum(
            torch.mean(
                torch.norm(
                    joints_input * 1000 - joints_output * 1000,
                    dim=3,
                ),
                dim=2,
            ),
            dim=0,
        )

        self.mpjpe += mpjpe.cpu().detach().numpy()

    def validation_test_end(self):
        self.mpjpe = self.mpjpe / self.nb_samples
        self.melv_input = self.melv_input / self.nb_samples
        self.melv_output = self.melv_output / self.nb_samples

    # edge_lengths: B x n_frames (25) x n_edges (20664)
    # edge_lengths_gt: B x 1 x n_edges (20664)
    def compute_edge_infos(self, edge_lengths, edge_lengths_gt, input_output):
        edge_variance = torch.abs(
            edge_lengths - edge_lengths_gt)  # B x 25 x 20664

        mev_mean = torch.mean(
            edge_variance,
            dim=2,
        )  # B x 25

        mev = torch.sum(
            mev_mean,
            dim=0,
        )  # 25

        mean_variance = (
            torch.mean(
                torch.flatten(edge_variance, 1, 2),
                dim=1,
            )
            .cpu()
            .numpy()
            .tolist()
        )

        if input_output is None:
            pass
        elif input_output == "input":
            self.all_mean_variance_input += mean_variance
            if hasattr(self, "melv_input"):
                self.melv_input += mev.cpu().detach().numpy()

            if not hasattr(self, "all_melv_input"):
                self.all_melv_input = mev_mean.clone()
            else:
                self.all_melv_input = torch.cat(
                    [self.all_melv_input, mev_mean], dim=0)
        elif input_output == "output":
            self.all_mean_variance_output += mean_variance
            if hasattr(self, "melv_output"):
                self.melv_output += mev.cpu().detach().numpy()

            if not hasattr(self, "all_melv_output"):
                self.all_melv_output = mev_mean.clone()
            else:
                self.all_melv_output = torch.cat(
                    [self.all_melv_output, mev_mean], dim=0
                )
        else:
            print("wrong input_output")
            exit()
