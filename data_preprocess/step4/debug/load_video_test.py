# from decord import VideoReader, cpu
from torchvision import io
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_pil_image
import importlib
if importlib.util.find_spec("torch_npu") is not None:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
else:
    torch_npu = None

import os
import gc
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from typing import List
from decord import VideoReader




def load_video_with_cut(video_file, cut=None, crop=None):
    # Read video using torchvision.io.read_video
    # cut video
    video_frames, audio, info = io.read_video(
        video_file,
        pts_unit="sec",
        output_format="TCHW",
    )

    # crop video
    s_x, e_x, s_y, e_y = crop
    video_frames = video_frames[cut[0]:cut[1], :, s_y:e_y, s_x:e_x]


    

    # sample video
    # assert info["video_fps"] == video_fps
    fps = info["video_fps"]
    video_length = video_frames.shape[0]
    duration = video_length / fps
    
    num_sample_frames = int(video_length/fps*3) 
    num_sample_frames_torchvision = int(video_length/video_fps*3) 

    indices = np.linspace(0, video_length - 1, num_sample_frames, dtype=int)
    indices_torchvision = np.linspace(0, video_length - 1, num_sample_frames_torchvision, dtype=int)

    

    import ipdb;ipdb.set_trace()

    video_frames = video_frames[indices]
    frame_timestamps = [int(duration / num_sample_frames * (i+0.5)) / fps for i in range(num_sample_frames)]

    return [to_pil_image(frame) for frame in video_frames], frame_timestamps

def get_video_info(video_path):
    """使用 decord 读取视频的帧率 (FPS)"""
    vr = VideoReader(video_path)  # 加载视频
    fps = vr.get_avg_fps()  # 获取 FPS
    first_frame = vr[0].asnumpy()  # 获取第一帧 (H, W, C)
    height, width = first_frame.shape[:2]  # 提取高度和宽度
    return fps, height, width


video_file = "/storage/hxy/ID/Consis-multiID/data_preprocess/step4/debug/1.mp4"

fps, height, width = get_video_info(video_file)


cut = [34, 227]
# fps = 23.979999542236328
crop = [0, width, 0, height]
frames, frame_timestamps = load_video_with_cut(video_file, fps, cut, crop)