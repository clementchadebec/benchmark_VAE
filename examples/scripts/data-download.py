import sys
from torchvision.datasets import MNIST, CelebA, CIFAR10
import argparse
from pathlib import Path
import torch
import numpy as np
from torchvision.transforms import PILToTensor
from tqdm import tqdm


def main():

    parser = argparse.ArgumentParser(
        description="python script to download datasets which are available with torchvision"
    )

    parser.add_argument(
        "-j", "--nthreads", type=int, default=1, help="number of threads to use"
    )
    parser.add_argument(
        "-b", "--batchsize", type=int, default=64, help="batch_size for loading"
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=Path("."),
        help="the base folder in which to store the output",
    )
    parser.add_argument(
        "dataset",
        nargs="+",
        help="datasets to download (possible values: MNIST, CelebA, CIFAR10)",
    )
    args = parser.parse_args()
    if not "dataset" in args:
        print("dataset argument not found in", args)
        parser.print_help()
        return 1

    tv_datasets = {"mnist": MNIST, "celeba": CelebA, "cifar10": CIFAR10}
    rootdir = args.outdir
    if not rootdir.exists():
        print(f"creating root folder {rootdir}")
        rootdir.mkdir(parents=True)

    for dname in args.dataset:
        if dname.lower() not in tv_datasets.keys():
            print(f"{dname} not available for download yet. skipping.")
            continue

        dfolder = rootdir / dname
        dataset = tv_datasets[dname]
        if "celeba" in dname.lower():
            train_kwarg = {"split": "train"}
            val_kwarg = {"split": "val"}
        else:
            train_kwarg = {"train": True}
            val_kwarg = {"train": False}

        train_data = dataset(
            dfolder, download=True, transform=PILToTensor(), **train_kwarg
        )
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=4, shuffle=False, num_workers=args.nthreads
        )

        train_batches = []
        for b, (x, y) in enumerate(tqdm(train_loader)):
            train_batches.append(x.clone().detach().numpy())

        val_data = dataset(dfolder, download=True, transform=PILToTensor(), **val_kwarg)
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=4, shuffle=True, num_workers=args.nthreads
        )
        val_batches = []
        for b, (x, y) in enumerate(tqdm(val_loader)):
            val_batches.append(x.clone().detach().numpy())

        train_x = np.concatenate(train_batches)
        np.savez_compressed(dfolder / "train_data.npz", data=train_x)
        print(
            "Wrote ",
            dfolder / "train_data.npz",
            f"(shape {train_x.shape}, {train_x.dtype})",
        )
        val_x = np.concatenate(val_batches)
        np.savez_compressed(dfolder / "eval_data.npz", data=val_x)
        print(
            "Wrote ", dfolder / "eval_data.npz", f"(shape {val_x.shape}, {val_x.dtype})"
        )

    return 0


if __name__ == "__main__":
    rv = main()
    sys.exit(rv)
