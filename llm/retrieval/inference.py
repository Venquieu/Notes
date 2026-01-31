import argparse
import logging
import os
import time

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from toolbox.filesystem import make_dataset
from toolbox.utils import run_parallel


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--ckpt", required=True, help="REQUIRED. Path to the checkpoint file"
    )
    parser.add_argument("-d", "--dir", required=True, help="REQUIRED. Image directory")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="REQUIRED. Direction to save the results (.npy)",
    )
    parser.add_argument(
        "-r", "--rows", type=int, default=[], nargs="*", help="List of rows to process"
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=16,
        type=int,
        help="Batch size for inference. Default: 16",
    )
    parser.add_argument(
        "-w",
        "--num_workers",
        default=8,
        type=int,
        help="Number of workers for data loading. Default: 8",
    )
    return parser


class InferenceDataset(Dataset):
    def __init__(
        self, img_dir: str, start_idx: int = 0, end_idx: int = None, size: int = None
    ):
        self.img_dir = img_dir
        self.size = size
        self.img_files = make_dataset(img_dir)
        end_idx = (
            len(self.img_files)
            if end_idx is None
            else min(end_idx, len(self.img_files))
        )
        self.img_files = self.img_files[start_idx:end_idx]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]

        img = Image.open(img_file)
        if self.size is not None:
            img = img.resize((self.size, self.size))

        return img_file, img


def collate_fn(batch: list):
    outputs = {
        "img_path": [item[0] for item in batch],
        "img": [dict(image=item[1]) for item in batch],
    }
    return outputs


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d | %(levelname)s | %(module)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = make_parser().parse_args()

    logging.info("Loading model from checkpoint: %s" % args.ckpt)
    model = SentenceTransformer(args.ckpt).bfloat16()
    logging.info(model)

    if len(args.rows) == 0:
        args.rows = [0, None]
    elif len(args.rows) == 1:
        args.rows.append(None)

    dataset = InferenceDataset(args.dir, args.rows[0], args.rows[1])
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
        shuffle=False,
        collate_fn=collate_fn,
    )

    logging.info("Output to: %s" % args.output)
    if len(args.output.split("/")) > 1:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for idx, item in enumerate(tqdm(dataloader)):
        # print(item["img_base64"])
        ts = time.time()
        embeds = model.encode(
            item["img"],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        tea = time.time()
        logging.info("Time to encode: %.2fs" % (tea - ts))
        logging.info(
            "Shape of embeddings: %s" % str(embeds.shape)
        )  # (batch_size, embed_dim), embed_dim=1536

        ts = time.time()
        embeds = embeds.cpu().numpy()
        arg_list = []
        for path, embed in zip(item["img_path"], embeds):
            file_name = os.path.splitext(os.path.basename(path))[0] + ".npy"
            save_path = os.path.join(args.output, file_name)
            arg_list.append((save_path, embed))

        run_parallel(np.save, arg_list, max_workers=args.batch_size)
        teb = time.time()
        logging.info("Time to write: %.2fs" % (teb - ts))
    logging.info("Inference completed.")
