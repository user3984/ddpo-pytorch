import os
import torch
import random
import pickle
import functools
from PIL import Image
from pycocotools import mask
from torchvision import transforms
from typing import Dict, List, Tuple
from torch.utils.data import Dataset
from torchmetrics.multimodal.clip_score import CLIPScore


"""
PLEASE ENSURE FOLLOWING FILES EXIST:
ddpo_pytorch/assets/high_freq_obj.txt
ddpo_pytorch/assets/other_obj.txt
ddpo_pytorch/assets/prompts.pkl
ddpo_pytorch/assets/training_img.pkl

"""

ASSET_PATH = "ddpo_pytorch/assets"
IMGSET_PATH = "train2017"
CLIP_MODEL = "openai/clip-vit-base-patch16"

clip_score_fn = CLIPScore(model_name_or_path=CLIP_MODEL)


@functools.lru_cache(maxsize=1)
def load_img_categories() -> Dict[int, str]:
    """
    data[3] = "airplane"
    """
    data = {}
    for fname in ["high_freq_obj", "other_obj"]:
        path = os.path.join(ASSET_PATH, f"{fname}.txt")
        with open(path, "r") as fp:
            for line in fp.readlines():
                tokens = line.split()
                ctg_id = tokens[0]
                ctg_description = tokens[1:]
                data[int(ctg_id)] = ctg_description
    return data


@functools.lru_cache(maxsize=1)
def load_prompts() -> Dict[int, List[Tuple]]:
    """
    prompts[698] = [('transform the monitor into a vintage 1950s television set',
                    'a vintage-style 1950s television set'),
                    ("make the monitor look like it's made entirely of transparent glass",
                    'a transparent glass monitor'),
                    ...]
    """
    path = os.path.join(ASSET_PATH, f"prompts.pkl")
    with open(path, 'rb') as fp:
        data = pickle.load(fp)
    return data


@functools.lru_cache(maxsize=1)
def load_img_metadata() -> Dict[int, List[Dict]]:
    """
    data[432732] = [{
        'bbox': [91.01, 89.24, 388.85, 268.92],
        'category_id': 1202,
        'image_id': 432732,
        'id': 1,
        'segmentation': [
            [327.26, ..., 354.04],
            [372.23, ..., 230.42]
        ],
        'area': 46323.61
    }]
    """
    path = os.path.join(ASSET_PATH, f"training_img.pkl")
    with open(path, 'rb') as fp:
        data = pickle.load(fp)
    return data


class EditingDataset(Dataset):
    def __init__(self):
        self.img_to_meta = load_img_metadata()
        self.ctg_to_prompts = load_prompts()
        self.img_ids = list(self.img_to_meta.keys())
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.img_to_meta)

    def __getitem__(self, index):
        img = None
        prompt = None
        prompt_score_token = None
        bin_mask = None
        bbox = None

        img_id = self.img_ids[index]
        anno = random.choice(self.img_to_meta[img_id])
        ctg_id = anno["category_id"]

        # image
        path = os.path.join(IMGSET_PATH, f"{img_id:012}.jpg")
        img = Image.open(path)

        # prompt
        prompt, prompt_score_token = random.choice(self.ctg_to_prompts[ctg_id])

        # bin_mask
        w, h = img.size
        segs = anno["segmentation"]
        rle = mask.frPyObjects(segs, h, w)
        bin_mask = mask.decode(rle)

        # bbox
        bbox = anno["bbox"]

        # convert to tensors
        img = self.transform(img)
        bin_mask = torch.from_numpy(bin_mask)
        bbox = torch.tensor(bbox)

        return img, prompt, prompt_score_token, bin_mask, bbox

