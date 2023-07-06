import numpy as np
import torch
import imgui
import time

import learning.trainer as trainer

from aitviewer.renderables.meshes import Meshes
from aitviewer.viewer import Viewer
from visualization.colors import color_0, color_1


def upsample_anim(vertices, upsample_layer):
    vertices = vertices.permute(2, 1, 0)
    vertices = upsample_layer(vertices)
    vertices = vertices.permute(2, 1, 0)
    return vertices


class CustomViewer(Viewer):
    def __init__(self):
        super().__init__()

        self.index = "0"
        self.framerate = 60
        self.offset = False
        self.train_test = False
        self.gui_controls["model"] = self.gui_model

        _, self.opt = trainer.load_trainer(load=True)

        scale_factor = self.framerate / self.opt["framerate"]

        self.upsample_layer = torch.nn.Upsample(
            scale_factor=scale_factor, mode="linear")

        self.permute_z_up = [0, 1, 2]
        self.opt["evecs"] = self.opt["evecs"][:, : self.opt["nb_freqs"]]
        self.opt["model"].eval()
        self.load_dataset()
        self.load_sample()

        self.meshes_input = Meshes(
            self.vertices_input,
            self.opt["faces"],
            name="input meshes",
            color=color_0,
            flat_shading=False
        )

        index_start_output = 120

        multiple_colors = np.zeros((180, 6890, 4))
        multiple_colors[:index_start_output] = color_0
        multiple_colors[index_start_output:] = color_1

        self.meshes_output = Meshes(
            self.vertices_output,
            self.opt["faces"],
            name="output meshes",
            vertex_colors=multiple_colors,
            flat_shading=False
        )

        self.set_vertices()

    def gui_model(self):
        imgui.set_next_window_position(
            50, 100 + self.window_size[1] * 0.7, imgui.FIRST_USE_EVER
        )
        imgui.set_next_window_size(
            self.window_size[0] * 0.4, self.window_size[1] *
            0.15, imgui.FIRST_USE_EVER
        )
        expanded, _ = imgui.begin("SpecTrHuMS UI", None)

        if expanded:
            u, self.opt["job_id"] = imgui.input_text(
                "job_id",
                self.opt["job_id"],
                12,
                imgui.INPUT_TEXT_CHARS_DECIMAL | imgui.INPUT_TEXT_AUTO_SELECT_ALL,
            )

            u, self.train_test = imgui.checkbox(
                "Train (checked) or Test (unchecked) dataset", self.train_test)
            if u:
                self.load_dataset()

            _, self.index = imgui.input_text(
                "Index sample / {}".format(self.dataloader.__len__()),
                self.index,
                12,
                imgui.INPUT_TEXT_CHARS_DECIMAL | imgui.INPUT_TEXT_AUTO_SELECT_ALL,
            )

            if imgui.button("Load"):
                self.load_sample()
                self.set_vertices()

            if imgui.button("Random"):
                np.random.seed(seed=int(time.time()))

                length_dataset = self.dataloader.__len__()
                self.index = str(np.random.randint(0, length_dataset))

                self.load_sample()
                self.set_vertices()

            u, self.offset = imgui.checkbox("Offset", self.offset)

            if u:
                self.load_sample()
                self.set_vertices()

        self.prevent_background_interactions()
        imgui.end()

    def add_meshes(self):
        self.scene.add(self.meshes_input)
        self.scene.add(self.meshes_output)

    def load_dataset(self):
        self.dataloader = (
            self.opt["dataloader_train"].dataset
            if self.train_test
            else self.opt["dataloader_test"].dataset
        )

    def load_sample(self):
        self.scene.current_frame_id = 0

        coeffs_input, _, _, _ = self.dataloader.__getitem__(int(self.index))

        coeffs_input = (
            torch.tensor(coeffs_input).to(
                self.opt["device"]).view(-1, self.opt["nb_freqs"], 3)
        )

        self.coeffs_input = coeffs_input

        self.infer()

    def infer(self):
        with torch.no_grad():
            self.coeffs_output = self.opt["model"].forward(self.coeffs_input)

        self.get_vertices()

    def get_vertices(self):
        self.coeffs_input = upsample_anim(
            self.coeffs_input, self.upsample_layer)
        self.coeffs_output = upsample_anim(
            self.coeffs_output, self.upsample_layer)

        vertices_input = torch.matmul(self.opt["evecs"], self.coeffs_input)
        vertices_output = torch.matmul(self.opt["evecs"], self.coeffs_output)

        if self.offset:
            vertices_input[:, :, 0] -= 0.5
            vertices_output[:, :, 0] += 0.5

        self.vertices_input = vertices_input.cpu().numpy()
        self.vertices_output = vertices_output.cpu().numpy()

    def set_vertices(self):
        self.meshes_input.vertices = self.vertices_input
        self.meshes_output.vertices = self.vertices_output
