from zhipuai import ZhipuAI

import base64
import json
import decord
import numpy as np
from PIL import Image
import io
import argparse
import os

video_path = "/storage/hxy/ID/data/data_processor/test/wrong_examples/gm621059908-108728121_part2.mp4"
with open(video_path, 'rb') as video_file:
  video_base = base64.b64encode(video_file.read()).decode('utf-8')


def extract_and_crop_frames(video_path, start_frame, end_frame, top_left, bottom_right):
  """从视频中提取指定帧数的帧序列，并裁剪指定区域，返回Base64编码"""
  # 使用decord读取视频
  video_reader = decord.VideoReader(video_path)

  # 确保请求的帧在视频范围内
  start_frame = max(start_frame, 0)
  end_frame = min(end_frame, len(video_reader) - 1)

  # 提取连续帧
  frames = video_reader.get_batch(range(start_frame, end_frame + 1)).asnumpy()

  # 将所有帧整合成一个多维数组
  frames_array = np.concatenate(frames, axis=0)

  # 裁剪所有帧
  cropped_frames = frames_array[:, top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

  return cropped_frames

def frames_to_base64(frames):
  """将帧序列转换成Base64编码"""
  # 将所有帧合并成一个视频
  frames_list = [Image.fromarray(frame) for frame in frames]
  
  # 创建内存缓冲区
  buffer = io.BytesIO()
  
  # 将所有帧保存为一个视频
  frames_list[0].save(buffer, format='JPEG', save_all=True, append_images=frames_list[1:])
  
  # 将视频转换为Base64编码
  base64_encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
  
  return base64_encoded


 
def cap_video(video_base):

  prompt = ""

  client = ZhipuAI(api_key="388291dd58fd4e1ab75b43664825a712.PzOLjKEDTQrKWDHP") # 填写您自己的APIKey
  response = client.chat.completions.create(
      model="glm-4v-plus",  # 填写需要调用的模型名称
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
              "text": prompt
            }
          ]
        }
      ]
  )
  return response.choices[0].message

def parse_args():
  parser = argparse.ArgumentParser(description="Process MP4 files with YOLO models.")
  parser.add_argument('--input_json_folder', type=str, default='/storage/hxy/ID/data/data_processor/test/istockv1_extracted.json', help='Path to the folder containing MP4 files.')
  parser.add_argument('--output_json_folder', type=str, default='/storage/hxy/ID/data/data_processor/test/wrong_examples/frames_gm621059908', help='Directory for output files.')
  parser.add_argument('--video_root', type=str, default='/storage/hxy/ID/data/data_processor/test', help='Directory for output files.')
  parser.add_argument('--video_source', type=str, default='sucai', help='Directory for output files.')


  return parser.parse_args()

def main():
  args = parse_args()

  json_files = [f for f in os.listdir(args.input_json_folder) if f.endswith(".json")]


if __name__ == "__main__":
   main()