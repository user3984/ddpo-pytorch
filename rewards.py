from PIL import Image
import io
import numpy as np
import torch

from torchmetrics.multimodal.clip_score import CLIPScore


"""
PLEASE ENSURE FOLLOWING FILES EXIST:
ddpo_pytorch/assets/high_freq_obj.txt
ddpo_pytorch/assets/other_obj.txt
ddpo_pytorch/assets/prompts.pkl
ddpo_pytorch/assets/training_img.pkl

"""

ASSET_PATH = "assets"
IMGSET_PATH = "train2017"
CLIP_MODEL = "openai/clip-vit-base-patch16"
DEVICE = 'cuda'

clip_score_fn = CLIPScore(model_name_or_path=CLIP_MODEL).to(DEVICE)

@torch.no_grad()
def edit_reward(org_imgs, new_imgs, prompt_score_tokens, bin_masks, bboxes, w1=1000.0, w2=1.0, expand=0.1) -> float:
    """ reward = w1 * regional_penalty + w2 * semantic_similarity """

    regional_penalty = - _mse_outside_mask_batch(org_imgs, new_imgs, bin_masks)
    roi_imgs = _batchwise_crop(new_imgs, bboxes, expand)
    semantics_similarity = _semantics_similarity(roi_imgs, prompt_score_tokens)
    
    assert len(regional_penalty) == len(semantics_similarity)
    print("regional_penalty:", regional_penalty)
    print("semantics_similarity:", semantics_similarity)

    return w1 * regional_penalty + w2 * semantics_similarity


def _batchwise_crop(images, bboxes, expand=0.1):
    """
    Crop images in a batch according to specified bounding boxes.

    Parameters:
        images (torch.Tensor): Tensor of shape (B, C, H, W) where B is batch size,
                               C is number of channels, H is image height, and W is image width.
        bboxes (list of tuples): List of tuples (x, y, w, h) representing the bounding box for each image,
                                 where x, y are the coordinates of the top-left corner of the bounding box,
                                 w is the width, and h is the height of the bounding box.
        expand (float): expand factor for bounding box. Default=0.1.

    Returns:
        torch.Tensor: Tensor of cropped images. Note: Each image might have different sizes due to different bounding boxes.
    """
    cropped_images = []
    for image, (x, y, w, h) in zip(images, bboxes):
        # Ensure the bounding box is within the image dimensions
        img_h, img_w = image.shape[-2:]
        x, y, w, h = float(x), float(y), float(w), float(h)
        x_min = max(round(x - expand * w), 0)
        x_max = min(round(x + w + expand * w), img_w)
        y_min = max(round(y - expand * h), 0)
        y_max = min(round(y + h + expand * h), img_h)

        # Crop the image
        cropped_image = image[:, y_min:y_max, x_min:x_max]
        cropped_images.append(cropped_image)

    # Note: Returning a list because images can have different sizes
    return cropped_images

@torch.no_grad()
def calculate_clip_score(image, prompt):
    # print(image.shape)
    image_int = (image * 255).round().clamp(0, 255).to(torch.uint8)
    clip_score = clip_score_fn(image_int, prompt).detach()
    return float(clip_score)


def _semantics_similarity(images, prompt_score_tokens):
    scores = []
    for image, prompt in zip(images, prompt_score_tokens):
        scores.append(calculate_clip_score(image, prompt))

    return torch.tensor(scores).to(DEVICE)


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
    valid_squared_diff = (squared_diff * valid_mask).sum(dim=[1, 2, 3])
    mse = valid_squared_diff / valid_mask.sum(dim=[1, 2, 3])

    return mse

##########################################

def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew, meta

    return _fn


def aesthetic_score():
    from ddpo_pytorch.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


def llava_strict_satisfaction():
    """Submits images to LLaVA and computes a reward by matching the responses to ground truth answers directly without
    using BERTScore. Prompt metadata must have "questions" and "answers" keys. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 4
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadata_batched = np.array_split(metadata, np.ceil(len(metadata) / batch_size))

        all_scores = []
        all_info = {
            "answers": [],
        }
        for image_batch, metadata_batch in zip(images_batched, metadata_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [m["questions"] for m in metadata_batch],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            correct = np.array(
                [
                    [ans in resp for ans, resp in zip(m["answers"], responses)]
                    for m, responses in zip(metadata_batch, response_data["outputs"])
                ]
            )
            scores = correct.mean(axis=-1)

            all_scores += scores.tolist()
            all_info["answers"] += response_data["outputs"]

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn


def llava_bertscore():
    """Submits images to LLaVA and computes a reward by comparing the responses to the prompts using BERTScore. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 16
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        all_info = {
            "precision": [],
            "f1": [],
            "outputs": [],
        }
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [["Answer concisely: what is going on in this image?"]]
                * len(image_batch),
                "answers": [
                    [f"The image contains {prompt}"] for prompt in prompt_batch
                ],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            # use the recall score as the reward
            scores = np.array(response_data["recall"]).squeeze()
            all_scores += scores.tolist()

            # save the precision and f1 scores for analysis
            all_info["precision"] += (
                np.array(response_data["precision"]).squeeze().tolist()
            )
            all_info["f1"] += np.array(response_data["f1"]).squeeze().tolist()
            all_info["outputs"] += np.array(response_data["outputs"]).squeeze().tolist()

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn

