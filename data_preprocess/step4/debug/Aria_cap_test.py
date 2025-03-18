import os
import torch
import torchvision
from torchvision.io import read_video
from torchvision.transforms import functional as F
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from PIL import Image
import numpy as np
from typing import List

from vllm import LLM, SamplingParams

import requests
import torch
from PIL import Image
from transformers import AriaProcessor, AriaForConditionalGeneration, AutoTokenizer

temp = (
    "Please generate a comprehensive caption for the following video, describing various aspects, including but not limited to: "
    "1. The main theme and setting of the image (such as location, time of day, weather conditions, etc.) "
    "2. Key objects and their characteristics (such as color, shape, size, etc.) "
    "3. Relationships and interactions between objects (such as positioning, actions, etc.) "
    "4. Any people present and their emotions or activities (such as expressions, postures, etc.) "
    "5. Background and environmental details (such as architecture, natural scenery, etc.) "
    "6. Motion of the Subject: The movement of people or objects in the video. Use verbs that describe movement. "
    "7. Camera motion control: zoom in, zoom out, push in, pull out, pan right, pan left, truck right, truck left, tilt up, tilt down, pedestal up, pedestal down, arc shot,  tracking shot, static shot, and handheld shot. "
    'Do not describe imagined content. Only describe what can be determined from the video. Avoid listing things. Do not use abstract concepts (love, hate, justice, infinity, joy) as subjects. Use concrete nouns (human, cup, dog, planet, headphones) for more accurate results. Use verbs to describe the movement and changes of the subject or people. Write your prompts in plain, conversational language. Start your description directly with the main subject, typically a noun. Without "\n", subheading and title. '
    "Please describe the content of the video and the changes that occur, in chronological order:"
)



def load_video_with_cut(video_file, num_frames=256, video_fps=24, cut=None, crop=None, cache_dir="cached_video_frames", verbosity="DEBUG"):
    """
    读取视频文件，并仅提取指定时间段 (cut) 和裁剪指定区域 (crop)。

    :param video_file: 视频文件路径
    :param num_frames: 采样的帧数
    :param cut: (start_idx, end_idx) 选定帧的范围（单位：帧索引）
    :param crop: ((x1, y1), (x2, y2)) 选定裁剪区域（左上角坐标, 右下角坐标）
    :param cache_dir: 缓存目录
    :param verbosity: 设为 "DEBUG" 显示日志
    :return: (视频帧列表, 帧时间戳)
    """

    # Read video using torchvision.io.read_video
    video_frames, audio, info = read_video(
        video_file,
        start_pts=cut[0] / video_fps,
        end_pts=(cut[1] - 1)/ video_fps,
        pts_unit="sec",
        output_format="TCHW",
    )

    

    total_frames = video_frames.shape[0]
    fps = info["video_fps"]
    duration = total_frames / fps


    s_x, e_x, s_y, e_y = crop
    video_frames = video_frames[:, :, s_y:e_y, s_x:e_x]

    

    length = video_frames.shape[0]

    num_frames = int(length/fps*3) 

    indices = torch.linspace(0, length - 1, num_frames).long()  # 计算采样索引
    video_frames = video_frames[indices]
    frame_timestamps = [int(duration / num_frames * (i+0.5)) / fps for i in range(num_frames)]

    # import ipdb;ipdb.set_trace()

    return [to_pil_image(frame) for frame in video_frames], frame_timestamps



def get_placeholders_for_videos(frames: List, timestamps=[]):
    contents = []
    if not timestamps:
        for i, _ in enumerate(frames):
            contents.append({"text": None, "type": "image"})
        contents.append({"text": "\n", "type": "text"})
    else:
        for i, (_, ts) in enumerate(zip(frames, timestamps)):
            contents.extend(
                [
                    {"text": f"[{int(ts)//60:02d}:{int(ts)%60:02d}]", "type": "text"},
                    {"text": None, "type": "image"},
                    {"text": "\n", "type": "text"}
                ]
            )
    return contents



video_file = "/storage/hxy/ID/data/data_processor/sample/step4/cap_test/videos/gm621059908-108728121_part2.mp4"


# frames, frame_timestamps = load_video(video_file)

cut = [0,80]
crop = [0, 1000, 0, 600]



model_id_or_path = "/storage/hxy/ID/ckpts/Aria"

#Transformer
# model = AriaForConditionalGeneration.from_pretrained(model_id_or_path, device_map="auto", torch_dtype=torch.bfloat16)

# processor = AriaProcessor.from_pretrained(model_id_or_path)

#vllm
model = LLM(
        model=model_id_or_path,
        tokenizer=model_id_or_path,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 256},
        enforce_eager=True,
        trust_remote_code=True,
        max_model_len=38400,
        tensor_parallel_size=4,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.84,
    )

tokenizer = AutoTokenizer.from_pretrained(
        model_id_or_path, trust_remote_code=True, use_fast=False
    )

frames, frame_timestamps = load_video_with_cut(video_file=video_file, cut=cut, crop=crop)
contents = get_placeholders_for_videos(frames, frame_timestamps)


torch.cuda.empty_cache()

messages = [
    {
        "role": "user",
        "content": [
            *contents,
            {"text": temp, "type": "text"},
        ],
    }
]


#transformer
# text = processor.apply_chat_template(messages, add_generation_prompt=True)
# inputs = processor(text=text, images=frames, return_tensors="pt", max_image_size=490)
# inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
# inputs = {k: v.to(model.device) for k, v in inputs.items()}



# output = model.generate(
#     **inputs,
#     max_new_tokens=512,
#     stop_strings=["<|im_end|>"],
#     tokenizer=processor.tokenizer,
#     do_sample=True,
#     temperature=1,
# )
# output_ids = output[0][inputs["input_ids"].shape[1]:]
# result = processor.decode(output_ids, skip_special_tokens=True)


#vllm
text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

outputs = model.generate(
        {
            "prompt_token_ids": text,
            "multi_modal_data": {
                "image": frames,
                # "max_image_size": 490,  # [Optional] The max image patch size, default `980`
                # "split_image": False,  # [Optional] whether to split the images, default `False`
            },
        },
        sampling_params=SamplingParams(max_tokens=512, top_k=1, stop=["<|im_end|>"], temperature=1)
    )
generated_tokens = outputs[0].outputs[0].token_ids
result = tokenizer.decode(generated_tokens)

import ipdb;ipdb.set_trace()