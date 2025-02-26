import os
from typing import List, Optional, Tuple, Union

import cv2
import insightface
import numpy as np
import torch
from consisid_eva_clip import create_model_and_transforms
from consisid_eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from insightface.app import FaceAnalysis
from PIL import Image, ImageOps
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize
from transformers import T5EncoderModel, T5Tokenizer

from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.utils import load_image




def prepare_face_models(model_path, device, dtype):
    """
    Prepare all face models for the facial recognition task.

    Parameters:
    - model_path: Path to the directory containing model files.
    - device: The device (e.g., 'cuda', 'cpu') where models will be loaded.
    - dtype: Data type (e.g., torch.float32) for model inference.

    Returns:
    - face_helper_1: First face restoration helper.
    - face_helper_2: Second face restoration helper.
    - face_clip_model: CLIP model for face extraction.
    - eva_transform_mean: Mean value for image normalization.
    - eva_transform_std: Standard deviation value for image normalization.
    - face_main_model: Main face analysis model.
    """
    # get helper model
    face_helper_1 = FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model="retinaface_resnet50",
        save_ext="png",
        device=device,
        model_rootpath=os.path.join(model_path, "face_encoder"),
    )
    face_helper_1.face_parse = None
    face_helper_1.face_parse = init_parsing_model(
        model_name="bisenet", device=device, model_rootpath=os.path.join(model_path, "face_encoder")
    )
    face_helper_2 = insightface.model_zoo.get_model(
        f"{model_path}/face_encoder/models/antelopev2/scrfd_10g_bnkps.onnx", providers=["CUDAExecutionProvider"]
    )
    face_helper_2.prepare(ctx_id=0)

    # get local facial extractor part 1
    model, _, _ = create_model_and_transforms(
        "EVA02-CLIP-L-14-336",
        os.path.join(model_path, "face_encoder", "EVA02_CLIP_L_336_psz14_s6B.pt"),
        force_custom_clip=True,
    )
    face_clip_model = model.visual
    eva_transform_mean = getattr(face_clip_model, "image_mean", OPENAI_DATASET_MEAN)
    eva_transform_std = getattr(face_clip_model, "image_std", OPENAI_DATASET_STD)
    if not isinstance(eva_transform_mean, (list, tuple)):
        eva_transform_mean = (eva_transform_mean,) * 3
    if not isinstance(eva_transform_std, (list, tuple)):
        eva_transform_std = (eva_transform_std,) * 3
    eva_transform_mean = eva_transform_mean
    eva_transform_std = eva_transform_std

    # get local facial extractor part 2
    face_main_model = FaceAnalysis(
        name="antelopev2", root=os.path.join(model_path, "face_encoder"), providers=["CUDAExecutionProvider"]
    )
    face_main_model.prepare(ctx_id=0, det_size=(640, 640))

    # move face models to device
    face_helper_1.face_det.eval()
    face_helper_1.face_parse.eval()
    face_clip_model.eval()
    face_helper_1.face_det.to(device)
    face_helper_1.face_parse.to(device)
    face_clip_model.to(device, dtype=dtype)

    return face_helper_1, face_helper_2, face_clip_model, face_main_model, eva_transform_mean, eva_transform_std