from torch.utils.data import Dataset
import torch
import mmap
import numpy as np
import torch.nn.functional as F


def collate_fn(data):
    npmemmaps, genders, sizes_window, nbs_freqs = zip(*data)
    npmemmaps = np.array(npmemmaps).reshape(
        (len(data), sizes_window[0], nbs_freqs[0], 3))
    return torch.tensor(npmemmaps), torch.tensor(genders)


class CustomDataset(Dataset):
    def __init__(self, opt, train_test="train"):
        FLOAT_SIZE_np = 1
        self.nb_freqs = opt["nb_freqs"]
        self.SPECTRAL_COEFFS_SIZE_np = self.nb_freqs * 3 * FLOAT_SIZE_np

        filename_genders = opt["path_dataset"] + train_test + "_genders.bin"

        with open(filename_genders, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            self.genders = np.frombuffer(mm, dtype=int)

        filename_lengths = opt["path_dataset"] + train_test + "_lengths.bin"

        with open(filename_lengths, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            self.lengths = np.frombuffer(mm, dtype=int)

        self.sum_lengths = np.array(
            [np.sum(self.lengths[:i]) for i in range(len(self.lengths))]
        )

        self.filename_dataset = opt["path_dataset"] + train_test + ".bin"

        # in case another partition with faster write/read properties is available, for example on a supercomputer
        try:
            self.mm_dataset = np.memmap(
                "path_to_dataset_other_partition" + self.filename_dataset,
                dtype="float32",
                mode="r",
            )
            print(train_test + " dataloader, reading from faster partition partition")
        except:
            self.mm_dataset = np.memmap(
                self.filename_dataset, dtype="float32", mode="r"
            )
            print(train_test + " dataloader, reading from normal partition")

        self.nb_freqs = opt["nb_freqs"]

        self.size_window = opt["size_window"]
        self.offset = opt["offset"]

        self.batch_size = opt[train_test + "_batch_size"]

        self.build_indices()

        print(len(self.lengths), " anims")
        print(len(self.chunk_index_start_frame), " samples with offset")
        print()

    def __len__(self):
        # return 100  # uncomment for testing
        return len(self.chunk_index_start_frame)

    def __getitem__(self, idx):
        true_index, offset = self.chunk_index_start_frame[idx]

        return (
            self.mm_dataset[offset: offset +
                            self.size_window * self.nb_freqs * 3],
            self.genders[true_index],
            self.size_window,
            self.nb_freqs
        )

    def build_indices(self):
        self.chunk_index_start_frame = []

        for i in range(len(self.lengths)):
            current_length = self.lengths[i]
            for j in range(0, current_length, self.offset):
                if (j + self.size_window) < current_length:
                    offset = (self.sum_lengths[i] + j) * self.nb_freqs * 3
                    self.chunk_index_start_frame.append([i, offset])
