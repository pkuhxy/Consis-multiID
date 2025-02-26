import os
import pandas as pd

import cv2
from huggingface_hub import snapshot_download
from transformers import CLIPModel, CLIPProcessor


def compute_clip_score(video_path, model, processor, prompt, device, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []

    for i in range(num_frames):
        frame_idx = int(i * total_frames / num_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            print(f"Warning: Frame at index {frame_idx} could not be read.")

    cap.release()

    if model is not None:
        inputs = processor(text=prompt, images=frames, return_tensors="pt", padding=True, truncation=True)
        inputs.to(device)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        average_score = logits_per_image.mean().item()

    return average_score


# def main():
#     device = "cuda"
#     model_path = "/storage/hxy/ID/ckpts/consisID/data_process/clip-vit-base-patch32"
#     prompt = "The video depicts a young woman sitting at a wooden desk, deeply engrossed in her work. She is wearing glasses and has long hair that falls over her shoulders. The woman appears to be focused on a document or piece of paper in front of her, as she writes with a pen. Her posture suggests concentration and determination. The desk is cluttered with numerous stacks of papers and documents, indicating a busy and possibly overwhelming workload. The lighting in the room is dim, casting shadows and creating a somewhat somber atmosphere. The woman's expression is one of intense focus, with her hand resting on her chin, suggesting deep thought or contemplation. The overall scene conveys a sense of hard work and dedication, highlighting the challenges faced by individuals in managing their tasks and responsibilities. The presence of the large piles of paperwork emphasizes the volume of work that needs to be completed, adding to the pressure and stress associated with such tasks."
#     video_file_path = "/storage/hxy/ID/data/infer/vidu/0.mp4"
#     results_file_path = "/storage/hxy/ID/Consis-multiID/eval/clip_score/facesim_fid_score.txt"

#     if not os.path.exists(model_path):
#         print("Model not found, downloading from Hugging Face...")
#         snapshot_download(repo_id="openai/clip-vit-base-patch32", local_dir=model_path)
#     else:
#         print(f"Model already exists in {model_path}, skipping download.")

#     clip_model = CLIPModel.from_pretrained(model_path)
#     clip_processor = CLIPProcessor.from_pretrained(model_path)
#     clip_model.to(device)

#     clip_score = compute_clip_score(video_file_path, clip_model, clip_processor, prompt, device, num_frames=16)

#     # Write results to file
#     with open(results_file_path, 'w') as f:
#         f.write(f"clip score: {clip_score}\n")

#     # Print results
#     print(f"clip score: {clip_score}")


def main():
    device = "cuda"
    model_path = "/storage/hxy/ID/ckpts/consisID/data_process/clip-vit-base-patch32"
    video_folder = "/storage/hxy/ID/data/infer/vidu"  # mp4 文件所在文件夹
    prompt_file = "/storage/hxy/ID/data/ID/prompt.xlsx"  # 包含 prompt 的 Excel 文件路径
    results_file_path = "/storage/hxy/ID/Consis-multiID/eval/clip_score/vidu_score.txt"

    # 1. 读取 xlsx 文件中的 prompt
    df = pd.read_excel(prompt_file)
    prompt_dict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))  # 第一列是标号，第二列是 prompt

    # import ipdb; ipdb.set_trace()
    

    # 2. 加载 CLIP 模型
    if not os.path.exists(model_path):
        print("Model not found, downloading from Hugging Face...")
        snapshot_download(repo_id="openai/clip-vit-base-patch32", local_dir=model_path)
    else:
        print(f"Model already exists in {model_path}, skipping download.")

    clip_model = CLIPModel.from_pretrained(model_path)
    clip_processor = CLIPProcessor.from_pretrained(model_path)
    clip_model.to(device)

    # 3. 遍历文件夹中的 mp4 文件
    results = []
    for video_file in os.listdir(video_folder):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(video_folder, video_file)
            video_prefix = os.path.splitext(video_file)[0]  # 提取文件前缀名

            video_prefix = int(video_prefix)

            # import ipdb; ipdb.set_trace()

            # 4. 根据前缀名找到对应的 prompt
            if video_prefix in prompt_dict:
                prompt = prompt_dict[video_prefix]
                # 5. 计算 clipscore
                clip_score = compute_clip_score(video_path, clip_model, clip_processor, prompt, device, num_frames=16)
                results.append((video_prefix, clip_score))
                print(f"Video: {video_prefix}, CLIP Score: {clip_score}")
            else:
                print(f"No prompt found for video: {video_prefix}")

    # 6. 保存结果到文件
    with open(results_file_path, 'w') as f:
        for video_prefix, clip_score in results:
            f.write(f"{video_prefix}: {clip_score}\n")


if __name__ == "__main__":
    main()
