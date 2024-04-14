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
from torchmetrics.functional.multimodal import clip_score


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

clip_score_fn = functools.partial(clip_score, model_name_or_path=CLIP_MODEL)


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

    def __len__(self):
        return len(self.img_to_meta)

    def __getitem__(self, index):
        img = None
        prompt = None
        prompt_score_token = None
        bin_mask = None
        bbox = None

        meta = self.img_to_meta[index]
        img_id = meta["img_id"]
        ctg_id = meta["category_id"]

        # image
        path = os.path.join(IMGSET_PATH, f"{img_id:012}.jpg")
        img = transform(Image.open(path))

        # prompt
        prompt, prompt_score_token = random.choice(self.ctg_to_prompts[ctg_id])

        # bin_mask
        w, h = img.size
        segs = meta["segmentation"]
        rle = mask.frPyObjects(segs, h, w)
        bin_mask = mask.decode(rle)

        # bbox
        bbox = meta["bbox"]

        # convert to tensors
        transform = transforms.ToTensor()
        img = transform(img)
        bin_mask = torch.from_numpy(bin_mask)
        bbox = torch.tensor(bbox)

        return img, prompt, prompt_score_token, bin_mask, bbox


@torch.no_grad
def reward(org_imgs, new_imgs, prompt_score_tokens, bin_masks, bboxes, w1=1.0, w2=1.0) -> float:
    """ reward = w1 * regional_penalty + w2 * semantic_similarity """

    regional_penalty = - _mse_outside_mask_batch(org_imgs, new_imgs, bin_masks)
    roi_imgs = _batchwise_crop(new_imgs, bboxes)
    semantics_similarity = _semantics_similarity(roi_imgs, prompt_score_tokens)

    assert len(regional_penalty) == len(semantics_similarity)

    return w1 * regional_penalty + w2 * semantics_similarity


def _batchwise_crop(images, bboxes):
    """
    Crop images in a batch according to specified bounding boxes.

    Parameters:
        images (torch.Tensor): Tensor of shape (B, C, H, W) where B is batch size,
                               C is number of channels, H is image height, and W is image width.
        bboxes (list of tuples): List of tuples (x, y, w, h) representing the bounding box for each image,
                                 where x, y are the coordinates of the top-left corner of the bounding box,
                                 w is the width, and h is the height of the bounding box.

    Returns:
        torch.Tensor: Tensor of cropped images. Note: Each image might have different sizes due to different bounding boxes.
    """
    cropped_images = []
    for image, (x, y, w, h) in zip(images, bboxes):
        # Ensure the bounding box is within the image dimensions
        x_end = x + w
        y_end = y + h
        # Crop the image
        cropped_image = image[:, y:y_end, x:x_end]
        cropped_images.append(cropped_image)

    # Note: Returning a list because images can have different sizes
    return cropped_images


def _semantics_similarity(images, prompt_score_tokens):

    def calculate_clip_score(images, prompts):
        images_int = (images * 255).astype("uint8")
        clip_score = clip_score_fn(torch.from_numpy(
            images_int).permute(0, 3, 1, 2), prompts).detach()
        return round(float(clip_score), 4)

    scores = []
    for image, prompt in zip(images, prompt_score_tokens):
        scores.append(calculate_clip_score([image], [prompt]))

    return torch.tensor(scores)


def _mse_outside_mask_batch(tensorA, tensorB, tensor_mask):
    """
    Compute the MSE (Mean Squared Error) between batches of images outside the masked regions using PyTorch.

    Parameters:
        tensorA (torch.Tensor): First batch of images (tensor format, shape BxCxHxW).
        tensorB (torch.Tensor): Second batch of images (tensor format, shape BxCxHxW).
        tensor_mask (torch.Tensor): Batch of binary masks where regions to exclude are white (1) and regions to include are black (0).

    Returns:
        torch.Tensor: MSE values for each pair in the batch.
    """
    # Ensure mask is boolean (0 for regions to include)
    valid_mask = tensor_mask == 0

    # Ensure valid_mask covers all color channels
    valid_mask = valid_mask.unsqueeze(1).repeat(1, 3, 1, 1)

    # Calculate squared differences
    diff = tensorA - tensorB
    squared_diff = diff.pow(2)

    # Apply the mask and compute MSE
    valid_squared_diff = squared_diff[valid_mask].view(tensorA.size(0), -1)
    mse = valid_squared_diff.mean(dim=1)

    return mse
