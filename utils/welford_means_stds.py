import torch


class Welford:
    def __init__(self, dims, path, device):
        self.dims = dims
        self.dims_extended = tuple([2]) + dims
        self.path = path
        self.device = device

        self.initialize_values()

    def initialize_values(self):
        self.means_stds = torch.zeros(self.dims_extended).to(self.device)
        self.means_stds[1, :] = 1
        self.n = 0
        self.M2 = torch.zeros(self.dims).to(self.device)

    def aggregate(self, tensor, flatten=True):
        with torch.no_grad():
            if tensor.dim() == 3 and flatten:
                tensor = torch.flatten(tensor, 0, 1)

            self.n += len(tensor)
            delta = tensor - self.means_stds[0]
            sum_deltas = torch.sum(delta / self.n, 0)
            self.means_stds[0] = self.means_stds[0] + sum_deltas
            delta_2 = tensor - self.means_stds[0]
            delta_product = torch.sum(delta * delta_2, 0)
            self.M2 = self.M2 + delta_product

    def finalize(self):
        variance = self.M2 / (self.n - 1)
        self.means_stds[1] = variance.sqrt()

    def save(self):
        torch.save(
            self.means_stds.cpu(),
            self.path,
        )
