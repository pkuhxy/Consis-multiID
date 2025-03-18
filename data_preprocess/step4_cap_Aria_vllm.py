# from decord import VideoReader, cpu
import torch
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
from PIL import Image


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

    print(f"cut[0]:{cut[0]},cut[1]:{cut[1]}")

    # sample video
    fps = info["video_fps"]
    video_length = video_frames.shape[0]
    duration = video_length / fps
    # num_sample_frames = int(video_length/fps*3) 
    num_sample_frames = 12

    indices = np.linspace(0, video_length - 1, num_sample_frames, dtype=int)
    video_frames = video_frames[indices]
    # frame_timestamps = [int(duration / num_sample_frames * (i+0.5)) / fps for i in indices]
    frame_timestamps = [i / fps for i in indices]

    print(len(indices), indices, frame_timestamps)

    def resize_longest_edge(image):
        w, h = image.size
        if w > h:
            new_w, new_h = 512, int(512 * h / w)  # 宽 > 高
        else:
            new_w, new_h = int(512 * w / h), 512  # 高 > 宽
        return image.resize((new_w, new_h), Image.LANCZOS)  # **修正 LANCZOS 访问方式**

    # resized_images = [resize_longest_edge(to_pil_image(frame)) for frame in video_frames]

    images = [to_pil_image(frame) for frame in video_frames]

    return images, frame_timestamps



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


# temp = """
# Please describe the content of this video in as much detail as possible,
# including the objects, scenery, animals, characters, and camera movements within the video.
# Do not include '\n' in your response.
# Please start the description with the video content directly.
# Please describe the content of the video and the changes that occur, in chronological order.
# """
# temp = (
# "Please generate a comprehensive caption for the following video, describing various aspects, including but not limited to:"
# "1. The main theme and setting of the image (such as location, time of day, weather conditions, etc.)"
# "2. Key objects and their characteristics (such as color, shape, size, etc.)"
# "3. Relationships and interactions between objects (such as positioning, actions, etc.)"
# "4. Any people present and their emotions or activities (such as expressions, postures, etc.)"
# "5. Background and environmental details (such as architecture, natural scenery, etc.)"
# "6. Motion of the Subject: The movement of people or objects in the video. Use verbs that describe movement."
# "7. Camera motion control: zoom in, zoom out, push in, pull out, pan right, pan left, truck right, truck left, tilt up, tilt down, pedestal up, pedestal down, arc shot,  tracking shot, static shot, and handheld shot."
# "Do not describe imagined content. Only describe what can be determined from the video. Avoid listing things. Do not use abstract concepts (love, hate, justice, infinity, joy) as subjects. Use concrete nouns (human, cup, dog, planet, headphones) for more accurate results. Use verbs to describe the movement and changes of the subject or people. Write your prompts in plain, conversational language. Start your description directly with the main subject, typically a noun. Without \"\n\", subheading and title."
# "Please describe the content of the video and the changes that occur, in chronological order."
# )
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


class CaptionData(Dataset):
    def __init__(self, video_data, video_root, save_root, tokenizer):
        super().__init__()
        self.video_root = video_root
        self.save_root = save_root
        vid_paths = [i["path"] for i in video_data]
        crops = [i["crop"] for i in video_data]
        cuts = [i["cut"] for i in video_data]
        video_keys = [i["video_key"] for i in video_data]
        fpss = [i["fps"] for i in video_data]
        save_paths = [os.path.join(save_root, (i["video_key"]+".json")) for i in video_data]
        print("part x origin num", len(save_paths))
        self.paths = [[i, j, k, l, m, s] for i, j, k, l, m, s in zip(save_paths, vid_paths, crops, cuts, video_keys, fpss)]
        print("part x need to process num", len(self.paths))


        self.tokenizer = tokenizer
        self.executor = ThreadPoolExecutor(max_workers=1)

    def __len__(self):
        return len(self.paths)

    def load_video(self, path, crop, cut, fps):
        frames, frame_timestamps = load_video_with_cut(path, cut, crop)

        contents = get_placeholders_for_videos(frames, frame_timestamps)
        messages = [
            {
                "role": "user",
                "content": [
                    *contents,
                    {"text": temp, "type": "text"},
                ],
            }
        ]

        text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        inputs ={
            "prompt_token_ids": text,
            "multi_modal_data": {
                "image": frames,
            },
        }

        return inputs

    def wrapper(self, index):
        save_path, video_path, crop, cut, video_key, fps = self.paths[index]
        inputs = [self.load_video(video_path, crop, cut, fps)]
        return save_path, inputs, video_key

    def __getitem__(self, index):
        future = self.executor.submit(self.wrapper, index)
        save_path, inputs, video_key = future.result(timeout=50)
        return save_path, inputs, video_key
        # try:
        #     future = self.executor.submit(self.wrapper, index)
        #     save_path, inputs, video_key = future.result(timeout=50)
        #     return save_path, inputs, video_key
        # except Exception as e:
        #     print("error", e)
        #     return False, False, False


def collate_fn(batch):
    save_paths, inputs, video_key = zip(*batch)
    inputs = inputs[0]
    if not inputs:
        return False, False, False
    return save_paths, inputs, video_key


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_json", type=str, default="/storage/hxy/ID/data/data_processor/sample/step4/cap_test/jsons/gm621059908-108728121_part2_0-97.json")
    parser.add_argument("--video_root", type=str, default="/storage/hxy/ID/data/data_processor/sample")
    parser.add_argument("--save_root", type=str, default="/storage/hxy/ID/data/data_processor/sample/step4/Aria_test/cap_vllm_1.0")
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--total_part", type=int, default=1)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    num_workers = 0   # default = 10

    model_id_or_path = "/storage/hxy/ID/ckpts/Aria"
    # sampling_params = SamplingParams(
    #     repetition_penalty=1.05,
    #     temperature=1,
    #     # top_p=0.001,
    #     top_k=1,
    #     max_tokens=4096,
    #     stop=["<|im_end|>"]
    # )
    sampling_params = SamplingParams(
        temperature=1,
        max_tokens=512,
        top_k=1, 
        stop=["<|im_end|>"],
        # top_p=0.001,
        # repetition_penalty=1.05,
    )

    # default processer
    llm = LLM(
            model=model_id_or_path,
            tokenizer=model_id_or_path,
            dtype="bfloat16",
            limit_mm_per_prompt={"image": 256},
            enforce_eager=True,
            trust_remote_code=True,
            max_model_len=65536,
            tensor_parallel_size=4,
            distributed_executor_backend="mp",
            max_seq_len_to_capture=65536,
            # gpu_memory_utilization=0.84,
        )

    tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_path, trust_remote_code=True, use_fast=False
        )

    with open(args.input_json, "r") as f:
        origin_video_data = json.load(f)
    print("total data", len(origin_video_data))

    video_data = [
        {
            "path": i["path"],
            "video_key": key,
            "cut": i["face_cut"],
            "crop": i["crop"],
            "aes": i["aesthetic"],
            "tech": i["tech"],
            "motion": i["motion"],
            "fps": i["fps"],
            "num_frames": i["num_frames"],
            "resolution": [i["resolution"]["height"], i["resolution"]["width"]],
        }
        for key, i in tqdm(origin_video_data.items()) if len(i['face_cap_glm'])==0 and not os.path.exists(os.path.join(args.save_root, (key + ".json")))
    ]

    video_data_cp = {i["video_key"]: i for i in video_data}
    video_data = [
        {
            "path": os.path.join(args.video_root, i["path"]),
            "video_key": i["video_key"],
            "cut": i["cut"],
            "crop": i["crop"],
            "aes": i["aes"],
            "tech": i["tech"],
            "motion": i["motion"],
            "fps": i["fps"],
            "num_frames": i["num_frames"],
            "resolution": [i["resolution"][0], i["resolution"][1]],
        }
        for i in tqdm(video_data)
    ]

    # video_paths = list(glob(opj(args.video_root, '**', f'*.jpg'), recursive=True))
    print("after filter data", len(video_data))
    if args.shuffle:
        random.shuffle(video_data)
    video_data = video_data[args.part :: args.total_part]
    data = CaptionData(video_data, args.video_root, args.save_root, tokenizer)
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        num_workers=num_workers,
        # pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    for save_paths, frames, video_key in tqdm(loader):
        if not save_paths:
            continue
        folder, filename = os.path.split(save_paths[0])
        os.makedirs(folder, exist_ok=True)
        try:
            # results = []
            for inputs in frames:
                for i in range(100):
                # llm_inputs = inputs.to('cuda', non_blocking=True)
                # Inference
                    results = []
                    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):

                        import ipdb;ipdb.set_trace()

                        outputs = llm.generate([inputs], sampling_params=sampling_params)
                        generated_text = outputs[0].outputs[0].text
                        results.append(generated_text)

                    base_dict = origin_video_data[video_key[0]]
                    base_dict.update({"face_cap_Aria": results})

                    update_json = {}
                    update_json['metadata'] = base_dict

                    save_test_path = save_paths[0].replace('.json',f"_{i}.json")

                    print(save_test_path)
                    with open(save_test_path, "w") as f:
                        json.dump(update_json, f, indent=4)
        except Exception as e:
            print(e)

    print("Done")


"""

conda activate aes1
cd /storage/lcm/ocr
PART=96
SAVE_ROOT=cap_test_vllm/20241022
VIDEO_ROOT=/storage/dataset/xigua_sourcecode
JSON=/storage/lcm/video_spliting/results_recheck/20241022_3494978_aes_ocr_vq/final.json

CUDA_VISIBLE_DEVICES=1 python step4_cap_vllm.py --part 0 --total_part ${PART} --video_root ${VIDEO_ROOT} --save_root ${SAVE_ROOT} --json ${JSON}



conda activate aes1
cd /storage/lcm/ocr
PART=96
SAVE_ROOT=cap_test_vllm/20241022
VIDEO_ROOT=/storage/dataset/xigua_sourcecode
JSON=/storage/lcm/video_spliting/results_recheck/20241022_3494978_aes_ocr_vq/final.json
for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$((i % 8)) python step4_cap_vllm.py --part $i --total_part ${PART} --video_root ${VIDEO_ROOT} --save_root ${SAVE_ROOT} --json ${JSON} &
done
wait




conda activate aes1
cd /storage/lcm/ocr
PART=96
SAVE_ROOT=cap/20241022
VIDEO_ROOT=/storage/dataset/xigua_sourcecode
JSON=/storage/lcm/video_spliting/results_recheck/20241022_3494978_aes_ocr_vq/final.json
for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$((i % 8)) python step4_cap.py --part $i --total_part ${PART} --video_root ${VIDEO_ROOT} --save_root ${SAVE_ROOT} --json ${JSON} &
done
wait


conda activate aes1
cd /storage/lcm/ocr
PART=96
SAVE_ROOT=cap/20241022
VIDEO_ROOT=/storage/dataset/xigua_sourcecode
JSON=/storage/lcm/video_spliting/results_recheck/20241022_3494978_aes_ocr_vq/final.json
for i in {8..15}; do
    CUDA_VISIBLE_DEVICES=$((i % 8)) python step4_cap.py --part $i --total_part ${PART} --video_root ${VIDEO_ROOT} --save_root ${SAVE_ROOT} --json ${JSON} &
done
wait


conda activate aes1
cd /storage/lcm/ocr
PART=96
SAVE_ROOT=cap/20241022
VIDEO_ROOT=/storage/dataset/xigua_sourcecode
JSON=/storage/lcm/video_spliting/results_recheck/20241022_3494978_aes_ocr_vq/final.json
for i in {16..23}; do
    CUDA_VISIBLE_DEVICES=$((i % 8)) python step4_cap.py --part $i --total_part ${PART} --video_root ${VIDEO_ROOT} --save_root ${SAVE_ROOT} --json ${JSON} &
done
wait


conda activate aes1
cd /storage/lcm/ocr
PART=96
SAVE_ROOT=cap/20241022
VIDEO_ROOT=/storage/dataset/xigua_sourcecode
JSON=/storage/lcm/video_spliting/results_recheck/20241022_3494978_aes_ocr_vq/final.json
for i in {24..31}; do
    CUDA_VISIBLE_DEVICES=$((i % 8)) python step4_cap.py --part $i --total_part ${PART} --video_root ${VIDEO_ROOT} --save_root ${SAVE_ROOT} --json ${JSON} &
done
wait


conda activate aes1
cd /storage/lcm/ocr
PART=96
SAVE_ROOT=cap/20241022
VIDEO_ROOT=/storage/dataset/xigua_sourcecode
JSON=/storage/lcm/video_spliting/results_recheck/20241022_3494978_aes_ocr_vq/final.json
for i in {32..39}; do
    CUDA_VISIBLE_DEVICES=$((i % 8)) python step4_cap.py --part $i --total_part ${PART} --video_root ${VIDEO_ROOT} --save_root ${SAVE_ROOT} --json ${JSON} &
done
wait



conda activate aes1
cd /storage/lcm/ocr
PART=96
SAVE_ROOT=cap/20241022
VIDEO_ROOT=/storage/dataset/xigua_sourcecode
JSON=/storage/lcm/video_spliting/results_recheck/20241022_3494978_aes_ocr_vq/final.json
for i in {40..47}; do
    CUDA_VISIBLE_DEVICES=$((i % 8)) python step4_cap.py --part $i --total_part ${PART} --video_root ${VIDEO_ROOT} --save_root ${SAVE_ROOT} --json ${JSON} &
done
wait


conda activate aes1
cd /storage/lcm/ocr
PART=96
SAVE_ROOT=cap/20241022
VIDEO_ROOT=/storage/dataset/xigua_sourcecode
JSON=/storage/lcm/video_spliting/results_recheck/20241022_3494978_aes_ocr_vq/final.json
for i in {48..55}; do
    CUDA_VISIBLE_DEVICES=$((i % 8)) python step4_cap.py --part $i --total_part ${PART} --video_root ${VIDEO_ROOT} --save_root ${SAVE_ROOT} --json ${JSON} &
done
wait


conda activate aes1
cd /storage/lcm/ocr
PART=96
SAVE_ROOT=cap/20241022
VIDEO_ROOT=/storage/dataset/xigua_sourcecode
JSON=/storage/lcm/video_spliting/results_recheck/20241022_3494978_aes_ocr_vq/final.json
for i in {56..63}; do
    CUDA_VISIBLE_DEVICES=$((i % 8)) python step4_cap.py --part $i --total_part ${PART} --video_root ${VIDEO_ROOT} --save_root ${SAVE_ROOT} --json ${JSON} &
done
wait


conda activate aes1
cd /storage/lcm/ocr
PART=96
SAVE_ROOT=cap/20241022
VIDEO_ROOT=/storage/dataset/xigua_sourcecode
JSON=/storage/lcm/video_spliting/results_recheck/20241022_3494978_aes_ocr_vq/final.json
for i in {64..71}; do
    CUDA_VISIBLE_DEVICES=$((i % 8)) python step4_cap.py --part $i --total_part ${PART} --video_root ${VIDEO_ROOT} --save_root ${SAVE_ROOT} --json ${JSON} &
done
wait


conda activate aes1
cd /storage/lcm/ocr
PART=96
SAVE_ROOT=cap/20241022
VIDEO_ROOT=/storage/dataset/xigua_sourcecode
JSON=/storage/lcm/video_spliting/results_recheck/20241022_3494978_aes_ocr_vq/final.json
for i in {72..79}; do
    CUDA_VISIBLE_DEVICES=$((i % 8)) python step4_cap.py --part $i --total_part ${PART} --video_root ${VIDEO_ROOT} --save_root ${SAVE_ROOT} --json ${JSON} &
done
wait


conda activate aes1
cd /storage/lcm/ocr
PART=96
SAVE_ROOT=cap/20241022
VIDEO_ROOT=/storage/dataset/xigua_sourcecode
JSON=/storage/lcm/video_spliting/results_recheck/20241022_3494978_aes_ocr_vq/final.json
for i in {80..87}; do
    CUDA_VISIBLE_DEVICES=$((i % 8)) python step4_cap.py --part $i --total_part ${PART} --video_root ${VIDEO_ROOT} --save_root ${SAVE_ROOT} --json ${JSON} &
done
wait


conda activate aes1
cd /storage/lcm/ocr
PART=96
SAVE_ROOT=cap/20241022
VIDEO_ROOT=/storage/dataset/xigua_sourcecode
JSON=/storage/lcm/video_spliting/results_recheck/20241022_3494978_aes_ocr_vq/final.json
for i in {88..95}; do
    CUDA_VISIBLE_DEVICES=$((i % 8)) python step4_cap.py --part $i --total_part ${PART} --video_root ${VIDEO_ROOT} --save_root ${SAVE_ROOT} --json ${JSON} &
done
wait





ps -ef|grep step4_cap.py|grep -v grep|awk '{print $2}'|xargs kill -9

"""