from zhipuai import ZhipuAI

import base64
import json
import decord
import numpy as np
import argparse
import os
import cv2
import multiprocessing
from functools import partial



def save_video(frames, output_path, fps=30.0):
  """将裁剪后的帧保存为视频文件"""
  # 创建视频写入对象
  frame_height, frame_width, _ = frames[0].shape
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编解码器
  out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

  # 逐帧写入视频
  for frame in frames:
      out.write(frame.astype(np.uint8))  # 转换为uint8类型写入

  # 释放资源
  out.release()

  print(f"视频已保存到: {output_path}")


def extract_and_crop_frames(metadata, video_root):
  """从视频中提取指定帧数的帧序列，并裁剪指定区域，返回Base64编码"""
  # 使用decord读取视频
  video_path = os.path.join(video_root, metadata['path'])
  vr = decord.VideoReader(video_path)

  # 确保请求的帧在视频范围内
  cut = metadata['face_cut']

  # 提取连续帧
  frames = vr.get_batch(range(cut[0], cut[1])).asnumpy()

  # 裁剪所有帧
  s_x, e_x, s_y, e_y = metadata['crop']
  cropped_frames = frames[:, s_y:e_y, s_x:e_x, :]

  bgr_frames = np.array([cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in cropped_frames])

  return bgr_frames


def resize_frame(frame, target_long_side=512):
  """ 将帧的长边调整为 target_long_side，等比例缩放 """
  height, width, _ = frame.shape

  # 计算缩放比例
  scale = target_long_side / max(height, width)

  # 计算新尺寸（按比例缩放）
  new_width = int(width * scale)
  new_height = int(height * scale)

  # 调整大小
  resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
  
  return resized_frame


def save_frames_to_video_and_get_base64(frames, video_path):
  """ 将调整大小的帧序列保存为 MP4 视频 """
  if len(frames) == 0:
      print("No frames to save!")
      return

  # 获取原始帧的宽高
  height, width, _ = frames[0].shape
  target_long_side = 512  # 目标长边

  # 计算缩放后的新尺寸
  scale = target_long_side / max(height, width)
  new_width = int(width * scale)
  new_height = int(height * scale)

  # 初始化 VideoWriter
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
  video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (new_width, new_height))

  # 处理并写入帧
  for frame in frames:
    resized_frame = resize_frame(frame, target_long_side)  # 只进行等比例缩放
    video_writer.write(resized_frame)

  # 释放资源
  video_writer.release()

  # 读取视频文件并转换为Base64编码
  with open(video_path, "rb") as video_file:
      video_base64 = base64.b64encode(video_file.read()).decode('utf-8')

  # import ipdb;ipdb.set_trace()

  # 删除临时的视频文件
  os.remove(video_path)

  return video_base64


 
def cap_video(video_base, API):

  temp = (
  "Please generate a comprehensive caption for the following video, describing various aspects, including but not limited to: "
  "1. The main theme and setting of the image (such as location, time of day, weather conditions, etc.) "
  "2. Key objects and their characteristics (such as color, shape, size, etc.) "
  "3. Relationships and interactions between objects (such as positioning, actions, etc.) "
  "4. Any people present and their emotions or activities (such as expressions, postures, etc.) "
  "5. Background and environmental details (such as architecture, natural scenery, etc.) "
  "6. Motion of the Subject: The movement of people or objects in the video. Use verbs that describe movement. "
  "7. Camera motion control: zoom in, zoom out, push in, pull out, pan right, pan left, truck right, truck left, tilt up, tilt down, pedestal up, pedestal down, arc shot,  tracking shot, static shot, and handheld shot. "
  "Do not describe imagined content. Only describe what can be determined from the video. Avoid listing things. Do not use abstract concepts (love, hate, justice, infinity, joy) as subjects. Use concrete nouns (human, cup, dog, planet, headphones) for more accurate results. Use verbs to describe the movement and changes of the subject or people. Write your prompts in plain, conversational language. Start your description directly with the main subject, typically a noun. Without \"\n\", subheading and title. "
  "Please describe the content of the video and the changes that occur, in chronological order:"
  )

  client = ZhipuAI(api_key=API) # 填写您自己的APIKey
  response = client.chat.completions.create(
      model="glm-4v-plus-0111",  # 填写需要调用的模型名称
      messages=[
        {
          "role": "user",
          "content": [
            {
              "type": "video_url",
              "video_url": {
                  "url" : video_base
              }
            },
            {
              "type": "text",
              "text": temp
            }
          ]
        }
      ],
      max_tokens=512,
  )
  return response.choices[0].message

def parse_args():
  parser = argparse.ArgumentParser(description="Process MP4 files with YOLO models.")
  parser.add_argument('--input_json_folder', type=str, default='/storage/hxy/ID/data/data_processor/sample/step2/istockv4', help='Path to the folder containing MP4 files.')
  parser.add_argument('--output_json_folder', type=str, default='/storage/hxy/ID/data/data_processor/test/wrong_examples/frames_gm621059908', help='Directory for output files.')
  parser.add_argument('--video_root', type=str, default='/storage/dataset/istock/', help='Directory for output files.')
  parser.add_argument('--video_source', type=str, default='sucai', help='Directory for output files.')
  parser.add_argument('--threads', type=int, default=50, help='Directory for output files.')
  parser.add_argument('--API', type=str, default='da36c8ba202714f3c663577bf4c10c63.jfXshFLsTaHBavmz', help='Directory for output files.')
  return parser.parse_args()


def process_json_file_test(json_file, input_json_folder, output_json_folder, video_root):
  json_path = os.path.join(input_json_folder, json_file)
  update_json_path = os.path.join(output_json_folder, json_file)
  if os.path.exists(update_json_path):
    print(f"{update_json_path} is already exists")
    return 

  # 打开并加载JSON文件
  with open(json_path, 'r') as f:
      json_data = json.load(f)

  metadata = json_data['metadata']

  # 如果cut和face_cut相同，清空face_cap
  if metadata['cut'] == metadata['face_cut']:
      json_data['metadata']['face_cap'] = []
      with open(update_json_path, 'w') as f:
          json.dump(json_data, f, indent=4)
      return  # 跳过后续处理

  # 提取和裁剪帧
  crop_frames = extract_and_crop_frames(metadata, video_root)

  # 保存裁剪帧为视频并获取Base64编码
  temp_video_path = json_file.replace('.json','.mp4')
  video_base64 = save_frames_to_video_and_get_base64(crop_frames, temp_video_path)




def process_json_file(json_file, input_json_folder, output_json_folder, video_root, API):
  json_path = os.path.join(input_json_folder, json_file)
  update_json_path = os.path.join(output_json_folder, json_file)
  if os.path.exists(update_json_path):
    print(f"{update_json_path} is already exists")
    return 

  # 打开并加载JSON文件
  with open(json_path, 'r') as f:
      json_data = json.load(f)

  metadata = json_data['metadata']

  # 如果cut和face_cut相同，清空face_cap
  if metadata['cut'] == metadata['face_cut']:
      json_data['metadata']['face_cap'] = []
      with open(update_json_path, 'w') as f:
          json.dump(json_data, f, indent=4)
      return  # 跳过后续处理

  # 提取和裁剪帧
  crop_frames = extract_and_crop_frames(metadata, video_root)

  # 保存裁剪帧为视频并获取Base64编码
  temp_video_path = json_file.replace('.json','.mp4')
  video_base64 = save_frames_to_video_and_get_base64(crop_frames, temp_video_path)

  # 获取视频的描述
  video_caption = cap_video(video_base64, API)
  json_data['metadata']['face_cap'] = [video_caption.content]

  # 保存更新后的JSON文件
  
  with open(update_json_path, 'w') as f:
    json.dump(json_data, f, indent=4)
    print(f"{update_json_path} has been saved")



def process_json_files_in_parallel(json_files, input_json_folder, output_json_folder, video_root, API, num_workers=4):
  # 创建进程池
  with multiprocessing.Pool(num_workers) as pool:
    # 使用partial函数来固定参数
    func = partial(process_json_file, input_json_folder=input_json_folder, 
                    output_json_folder=output_json_folder, video_root=video_root, API=API)
    
    # 启动并行处理
    pool.map(func, json_files)



def main():
  args = parse_args()

  json_files = [f for f in os.listdir(args.input_json_folder) if f.endswith(".json")]

  os.makedirs(args.output_json_folder, exist_ok=True)

  # process_json_file_test(json_files[0], args.input_json_folder, args.output_json_folder, args.video_root)

  process_json_files_in_parallel(json_files, args.input_json_folder, args.output_json_folder, args.video_root, args.API, num_workers=args.threads)




if __name__ == "__main__":
   main()