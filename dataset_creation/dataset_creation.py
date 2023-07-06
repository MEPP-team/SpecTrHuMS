import torch
import numpy as np
import os
import mmap
import json
import utils.welford_means_stds as w

from os import path
from human_body_prior.body_model.body_model import BodyModel


splits = {
    "train": [
        "CMU",
        "MPI_Limits",
        "TotalCapture",
        "Eyes_Japan_Dataset",
        "KIT",
        "EKUT",
        "TCD_handMocap",
        "ACCAD",
    ],
    "test": ["BioMotionLab_NTroje"],
}

with open("dataset_creation/default_options_dataset.json", "r") as outfile:
    opt = json.load(outfile)

# evecs
print("loading evecs")
with open("data/evecs_4096.bin", "r+b") as f:
    mm = mmap.mmap(f.fileno(), 0)
    evecs = torch.tensor(np.frombuffer(
        mm[:], dtype=np.float32)).view(6890, 4096).to(opt["device"])

evecs = evecs[:, :opt["nb_freqs"]].transpose(0, 1)

# AMASS
print("loading body models")
num_betas = 16  # number of body parameters
num_dmpls = 8  # number of DMPL parameters

# male
bm_male = "data/smplh/male/model.npz"
dmpl_male = "data/dmpls/male/model.npz"

smplh_data = np.load(bm_male)
dmpl_data = np.load(dmpl_male)

print("loading male")
bm_male = BodyModel(
    bm_fname=bm_male,
    num_betas=num_betas,
    num_dmpls=num_dmpls,
    dmpl_fname=dmpl_male,
)

print("sending to gpu")
bm_male = bm_male.to(opt["device"])

# female
print("loading female")
bm_female = "data/smplh/female/model.npz"
dmpl_female = "data/dmpls/female/model.npz"

bm_female = BodyModel(
    bm_fname=bm_female,
    num_betas=num_betas,
    num_dmpls=num_dmpls,
    dmpl_fname=dmpl_female,
)

print("sending to gpu")
bm_female = bm_female.to(opt["device"])

os.makedirs(opt["path_dataset"], exist_ok=True)


def fill_dataset(train_test):
    print()
    print("Creating " + train_test + "...")
    print()

    print(os.getcwd())

    # compute means and stds only for training dataset
    if train_test == "train":
        welford = w.Welford(
            (opt["nb_freqs"], 3),
            opt["path_dataset"]
            + train_test
            + "_means_stds.pt",
        )

    total_nb_frames = 0
    total_nb_samples = 0

    path_dataset_file = opt["path_dataset"] + train_test + ".bin"
    path_lengths_file = opt["path_dataset"] + train_test + "_lengths.bin"
    path_genders_file = opt["path_dataset"] + train_test + "_genders.bin"

    # dataset file
    with \
            open(path_dataset_file, "wb") as dataset_file, \
            open(path_lengths_file, "wb") as lengths_file, \
            open(path_genders_file, "wb") as genders_file:

        datasets_names = os.listdir(opt["amass_directory"])

        for dataset_name in datasets_names:
            if dataset_name not in splits[train_test]:
                continue

            dataset_path = path.join(
                opt["amass_directory"], dataset_name)

            if not path.isdir(dataset_path):
                print('dataset path not found')
                exit()

            print(dataset_path)

            subjects_names = os.listdir(dataset_path)

            for subject_name in subjects_names:
                subject_path = path.join(dataset_path, subject_name)

                if not path.isdir(subject_path):
                    print('subject path not found: ', subject_path)
                    exit()

                print(subject_path)

                sequences_names = os.listdir(subject_path)

                for sequence_name in sequences_names:
                    if sequence_name == "shape.npz":
                        continue

                    sequence_path = path.join(
                        subject_path, sequence_name)

                    if not path.isfile(sequence_path):
                        print('sequence path not found: ', sequence_path)
                        exit()

                    npz_data = np.load(sequence_path)

                    subject_gender = npz_data["gender"]

                    if subject_gender == "male":
                        current_bm = bm_male
                        gender_array = np.array(0, dtype=int)
                    elif subject_gender == "female":
                        current_bm = bm_female
                        gender_array = np.array(1, dtype=int)
                    else:
                        print('neutral gender')
                        exit()

                    gender_array.tofile(genders_file)

                    mocap_framerate = npz_data["mocap_framerate"]

                    factor_framerate = int(
                        mocap_framerate / opt["framerate"])

                    if factor_framerate == 0:
                        print(
                            "Problem with factor_framerate with sequence_path: ", sequence_path)
                        exit()

                    new_npz_data = {}

                    for k, v in npz_data.items():
                        if k not in [
                            "root_orient",
                            "pose_body",
                            "pose_hand",
                            "trans",
                            "dmpls",
                            "poses",
                        ]:
                            new_npz_data[k] = npz_data[k]
                        else:
                            new_npz_data[k] = npz_data[k][::factor_framerate]
                            """ if k == "pose_body":
                                        new_npz_data[k][:, :3] = 0 """

                    time_length = len(new_npz_data["trans"])
                    total_nb_frames += time_length

                    # for gpu memory consumption, if the anim is too long
                    max_len = 500

                    length = 0

                    for i in range(0, len(new_npz_data["trans"][:]), max_len):
                        body_parms = {
                            # controls the body
                            "pose_body": torch.Tensor(
                                new_npz_data["poses"][i: i + max_len, 3:66]
                            ).to(opt["device"]),
                            # controls the finger articulation
                            "pose_hand": torch.Tensor(
                                new_npz_data["poses"][i: i + max_len, 66:]
                            ).to(opt["device"]),
                            # controls the body shape
                            "betas": torch.Tensor(
                                np.repeat(
                                    new_npz_data["betas"][:num_betas][
                                        np.newaxis
                                    ],
                                    repeats=len(
                                        new_npz_data["trans"][i: i + max_len]
                                    ),
                                    axis=0,
                                )
                            ).to(opt["device"]),
                        }

                        try:
                            body_pose_beta = current_bm(**body_parms)
                        except Exception as e:
                            print(e)
                            print("sequence_path: ", sequence_path)
                            print(
                                "len: ",
                                new_npz_data["poses"][
                                    i: i + max_len, 3:66
                                ].shape[0],
                            )

                        coeffs = torch.matmul(evecs, body_pose_beta.v)

                        if train_test == "train":
                            welford.aggregate(coeffs.clone(), flatten=False)

                        coeffs.cpu().numpy().tofile(dataset_file)

                        length += coeffs.shape[0]

                    total_nb_samples += 1

                    length = np.array(length, dtype=int)
                    length.tofile(lengths_file)

                    # uncomment if you want to create few samples for testing
                    '''if total_nb_samples > 100:
                        if train_test == "train":
                            welford.finalize()
                            welford.save()

                        return'''

    if train_test == "train":
        welford.finalize()
        welford.save()

    print("total_nb_frames", total_nb_frames)
    print("total_nb_samples", total_nb_samples)

    return


def create_dataset():
    for train_test in ["train", "test"]:
        fill_dataset(train_test)

    opt_created_dataset = {}
    opt_created_dataset["nb_freqs"] = opt["nb_freqs"]
    opt_created_dataset["framerate"] = opt["framerate"]

    with open(
        opt["path_dataset"] + "infos.json",
        "w",
    ) as outfile:
        json.dump(
            opt_created_dataset,
            outfile,
            sort_keys=True,
            indent=4,
        )
