import torch
import torch.nn as nn


class MatMulLayer(torch.nn.Module):
    def __init__(self, dim_0, dim_1) -> None:
        super().__init__()
        self.mat = torch.nn.Parameter(
            torch.zeros(dim_0, dim_1), requires_grad=True)

    def forward(self, x):
        return torch.matmul(x, self.mat)


class LearnedPooling(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()

        encoder_features = opt["features"]

        sizes_downsample = opt["lengths"]
        sizes_downsample[0] = opt["nb_freqs"]
        sizes_upsample = sizes_downsample[::-1]

        sizes_convs_encode = [opt["size_conv"]
                              for _ in range(len(encoder_features))]
        sizes_convs_decode = sizes_convs_encode[::-1]

        self.latent_space = opt["size_latent"]

        encoder_linear = [
            encoder_features[-1] * sizes_downsample[-1],
            self.latent_space,
        ]

        decoder_linear = [
            self.latent_space,
            encoder_features[-1] * sizes_downsample[-1],
        ]

        decoder_features = encoder_features[::-1]
        decoder_features[-1] = decoder_features[-2]

        self.size_view = sizes_downsample[-1]

        if opt["activation_autoencoder"] == "ReLU":
            self.activation = nn.ReLU
        elif opt["activation_autoencoder"] == "Tanh":
            self.activation = nn.Tanh
        elif opt["activation_autoencoder"] == "Sigmoid":
            self.activation = nn.Sigmoid
        elif opt["activation_autoencoder"] == "LeakyReLU":
            self.activation = nn.LeakyReLU
        elif opt["activation_autoencoder"] == "ELU":
            self.activation = nn.ELU
        else:
            print("Wrong activation")
            exit()

        # Encoder
        self.encoder_features = torch.nn.Sequential()

        for i in range(len(encoder_features) - 1):
            self.encoder_features.append(
                torch.nn.Conv1d(
                    encoder_features[i],
                    encoder_features[i + 1],
                    sizes_convs_encode[i],
                    padding=sizes_convs_encode[i] // 2,
                )
            )
            self.encoder_features.append(
                MatMulLayer(sizes_downsample[i], sizes_downsample[i + 1])
            )
            self.encoder_features.append(self.activation())

        self.encoder_linear = torch.nn.Sequential()

        for i in range(len(encoder_linear) - 1):
            self.encoder_linear.append(
                torch.nn.Linear(encoder_linear[i], encoder_linear[i + 1])
            )

        # Decoder
        self.decoder_linear = torch.nn.Sequential()

        for i in range(len(decoder_linear) - 1):
            self.decoder_linear.append(
                torch.nn.Linear(decoder_linear[i], decoder_linear[i + 1])
            )

        self.decoder_features = torch.nn.Sequential()

        for i in range(len(decoder_features) - 1):
            self.decoder_features.append(
                MatMulLayer(sizes_upsample[i], sizes_upsample[i + 1])
            )

            self.decoder_features.append(
                torch.nn.Conv1d(
                    decoder_features[i],
                    decoder_features[i + 1],
                    sizes_convs_decode[i],
                    padding=sizes_convs_decode[i] // 2,
                )
            )
            self.decoder_features.append(self.activation())

        self.last_conv = torch.nn.Conv1d(
            decoder_features[-1],
            3,
            sizes_convs_decode[-1],
            padding=sizes_convs_decode[-1] // 2,
        )

        self.means_stds = torch.load(
            opt["path_dataset"] + "train_means_stds.pt").to(opt["device"])

    def enc(self, x):

        # standardize
        x = x - self.means_stds[0]
        x = x / self.means_stds[1]

        x = x.permute(0, 2, 1)

        x = self.encoder_features(x)

        x = torch.flatten(x, start_dim=1, end_dim=2)

        x = self.encoder_linear(x)

        return x

    def dec(self, x):
        x = self.decoder_linear(x)

        x = x.view(x.shape[0], -1, self.size_view)

        x = self.decoder_features(x)

        x = self.last_conv(x)

        x = x.permute(0, 2, 1)

        # destandardize
        x = x * self.means_stds[1]
        x = x + self.means_stds[0]

        return x

    def forward(self, x):
        unsqueeze = False

        if x.dim() == 2:
            x = x.unsqueeze(0)
            unsqueeze = True

        latent = self.enc(x)
        output = self.dec(latent)

        if unsqueeze:
            output = output.squeeze(0)

        return output
