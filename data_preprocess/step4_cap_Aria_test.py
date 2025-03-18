import requests
import torch
from PIL import Image
from torchvision.io import read_video
from torchvision.transforms.functional import to_pil_image
from transformers import AriaProcessor, AriaForConditionalGeneration

# 加载 Aria 模型
model_id_or_path = "/storage/hxy/ID/ckpts/Aria"
device = "cuda"
model = AriaForConditionalGeneration.from_pretrained(
    model_id_or_path, device_map="auto", torch_dtype=torch.bfloat16
)
# model = AriaForConditionalGeneration.from_pretrained(
#     model_id_or_path, torch_dtype=torch.bfloat16
# ).to(device)

processor = AriaProcessor.from_pretrained(model_id_or_path)

def extract_frames(video_path, num_segments=16):
    """ 读取视频并均匀抽取 num_segments 帧 """
    video_frames, _, info = read_video(video_path, pts_unit='sec')
    total_frames = video_frames.shape[0]
    if total_frames == 0:
        raise ValueError("无法读取视频帧")
    
    fps = info["video_fps"]
    video_length = video_frames.shape[0]
    # num_segments = int(video_length/fps*3)     

    frame_indices = torch.linspace(0, total_frames - 1, num_segments).long()

    def resize_longest_edge(image):
        w, h = image.size
        if w > h:
            new_w, new_h = 512, int(512 * h / w)  # 宽 > 高
        else:
            new_w, new_h = int(512 * w / h), 512  # 高 > 宽
        return image.resize((new_w, new_h), Image.LANCZOS)  # **修正 LANCZOS 访问方式**

    frames = [Image.fromarray(video_frames[idx].numpy()) for idx in frame_indices]

    # resized_images = [resize_longest_edge(to_pil_image(frame)) for frame in frames]


    return frames

def infer_video(video_path, question, num_segments=8):
    frames = extract_frames(video_path, num_segments=num_segments)
    messages = [{
        "role": "user",
        "content": [{"type": "image"} for _ in frames] + [{"text": question, "type": "text"}],
    }]
    
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=frames, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
    inputs.to(model.device)
    
    output = model.generate(
        **inputs,
        max_new_tokens=2048,
        stop_strings=["<|im_end|>"],
        tokenizer=processor.tokenizer,
        do_sample=True,
        temperature=1,
    )
    output_ids = output[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(output_ids, skip_special_tokens=True)
    return response

# 示例：视频推理
video_path = "/storage/hxy/ID/data/data_processor/sample/step4/cap_test/videos/gm621059908-108728121_part2.mp4"
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


response = infer_video(video_path, temp)
print(f'Assistant: {response}')
