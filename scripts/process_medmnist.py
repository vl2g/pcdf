import os
import csv
import argparse
from PIL import Image
from tqdm import tqdm

from medmnist import (
    DermaMNIST,
    PathMNIST,
    PneumoniaMNIST,
    RetinaMNIST
)

DATASETS = {
    "dermamnist": {
        "class": DermaMNIST,
        "img_mode": "RGB",
        "labels": {
            '0': 'actinic keratoses',
            '1': 'basal cell carcinoma',
            '2': 'benign keratosis-like lesions',
            '3': 'dermatofibroma',
            '4': 'melanocytic nevi',
            '5': 'vascular lesions',
            '6': 'melanoma'
        },
        "labels_symp": None
    },
}

def to_pil(img, mode):
    if isinstance(img, Image.Image):
        return img.convert(mode)

    if img.ndim == 3:
        if img.shape[0] in [1, 3]:
            img = img.transpose(1, 2, 0)
        if img.shape[-1] == 1:
            img = img.squeeze(-1)

    return Image.fromarray(img).convert(mode)


def main(dataset_name):
    assert dataset_name in DATASETS, f"Unsupported dataset: {dataset_name}"

    cfg = DATASETS[dataset_name]
    DatasetClass = cfg["class"]

    data_root = f"MedMNIST/{DatasetClass.__name__}"
    csv_root = f"data/{dataset_name}"

    os.makedirs(data_root, exist_ok=True)
    os.makedirs(csv_root, exist_ok=True)

    datasets = {
        "train": DatasetClass(split="train", size=224, download=True, transform=None),
        # "val": DatasetClass(split="val", size=224, download=True, transform=None),
        "test": DatasetClass(split="test", size=224, download=True, transform=None),
    }

    for split, dataset in datasets.items():
        img_dir = os.path.join(data_root, split)
        os.makedirs(img_dir, exist_ok=True)

        csv_name = (
            "train.csv" if split == "train"
            else f"{dataset_name}_{split}.csv"
        )
        csv_path = os.path.join(csv_root, csv_name)

        fieldnames = ["image", "diagnosis"]
        if cfg["labels_symp"]:
            fieldnames.append("diagnosis_symp")

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for idx, (img, label_arr) in enumerate(
                tqdm(dataset, desc=f"{dataset_name}:{split}")
            ):
                label = str(label_arr.item())
                filename = f"{idx:05d}.png"

                img_pil = to_pil(img, cfg["img_mode"])
                img_pil.save(os.path.join(img_dir, filename))

                row = {
                    "image": filename,
                    "diagnosis": cfg["labels"][label]
                }

                if cfg["labels_symp"]:
                    row["diagnosis_symp"] = cfg["labels_symp"][label]

                writer.writerow(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=DATASETS.keys(),
        help="Dataset to preprocess"
    )
    args = parser.parse_args()
    main(args.dataset)