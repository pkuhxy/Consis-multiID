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


def parse_args():
  parser = argparse.ArgumentParser(description="Process MP4 files with YOLO models.")
  parser.add_argument('--input_json_folder', type=str, default='/storage/hxy/ID/data/data_processor/sample/step2/istockv4', help='Path to the folder containing MP4 files.')
  parser.add_argument('--output_json_folder', type=str, default='/storage/hxy/ID/data/data_processor/test/wrong_examples/frames_gm621059908', help='Directory for output files.')
  parser.add_argument('--video_root', type=str, default='/storage/dataset/istock/', help='Directory for output files.')
  parser.add_argument('--video_source', type=str, default='sucai', help='Directory for output files.')
  parser.add_argument('--threads', type=int, default=50, help='Directory for output files.')
  parser.add_argument('--API', type=str, default='da36c8ba202714f3c663577bf4c10c63.jfXshFLsTaHBavmz', help='Directory for output files.')
  return parser.parse_args()


def save_video(frames, output_path, fps=30.0):
  frame_height, frame_width, _ = frames[0].shape
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

  for frame in frames:
    out.write(frame.astype(np.uint8))

  out.release()

  print(f"video has been saved at: {output_path}")


def extract_and_crop_frames(metadata, video_root):
  video_path = os.path.join(video_root, metadata['path'])
  vr = decord.VideoReader(video_path)

  cut = metadata['face_cut']

  frames = vr.get_batch(range(cut[0], cut[1])).asnumpy()

  s_x, e_x, s_y, e_y = metadata['crop']
  cropped_frames = frames[:, s_y:e_y, s_x:e_x, :]

  bgr_frames = np.array([cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in cropped_frames])

  return bgr_frames


def resize_frame(frame, target_long_side=512):
  height, width, _ = frame.shape

  scale = target_long_side / max(height, width)

  new_width = int(width * scale)
  new_height = int(height * scale)

  resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
  
  return resized_frame


def save_frames_to_video_and_get_base64(frames, video_path, num_frames, fps):

  if len(frames) == 0:
    print("No frames to save!")
    return

  total_frames = len(frames)
  
  if total_frames < num_frames:
    print(f"warning: total_frames:{total_frames} < num_frames:{num_frames}!")
    selected_indices = np.arange(total_frames)
  else:
    selected_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)

  height, width, _ = frames[0].shape
  target_long_side = 512

  scale = target_long_side / max(height, width)
  new_width = int(width * scale)
  new_height = int(height * scale)

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  video_writer = cv2.VideoWriter(video_path, fourcc, fps/3, (new_width, new_height))
  
  # for i in selected_indices:
  #   resized_frame = resize_frame(frames[i], target_long_side)
  #   video_writer.write(resized_frame)

  for frame in frames:
    resized_frame = resize_frame(frame, target_long_side)
    video_writer.write(resized_frame)

  print(f"{video_path} has been saved")

  video_writer.release()

  with open(video_path, "rb") as video_file:
    video_base64 = base64.b64encode(video_file.read()).decode('utf-8')

  # os.remove(video_path)

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

  client = ZhipuAI(api_key=API) 
  response = client.chat.completions.create(
      model="glm-4v-plus-0111", 
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
      temperature=0.1,
      max_tokens=512,
  )
  return response.choices[0].message


def process_json_file_test(json_file, input_json_folder, output_json_folder, video_root, API):
  json_path = os.path.join(input_json_folder, json_file)
  update_json_path = os.path.join(output_json_folder, json_file)
  # if os.path.exists(update_json_path):
  #   print(f"{update_json_path} is already exists")
  #   return 

  with open(json_path, 'r') as f:
    json_data = json.load(f)

  metadata = json_data['metadata']

  import ipdb;ipdb.set_trace()

  # if metadata['cut'] == metadata['face_cut']:
  #     json_data['metadata']['face_cap'] = []
  #     with open(update_json_path, 'w') as f:
  #         json.dump(json_data, f, indent=4)
  #     return 

  video_root = ''
  crop_frames = extract_and_crop_frames(metadata, video_root)

  num_frames = 8

  temp_video_path = json_file.replace('.json','.mp4')
  video_base64 = save_frames_to_video_and_get_base64(crop_frames, temp_video_path, num_frames, int(metadata['fps']))
  video_caption = cap_video(video_base64, API)

  output_json_path = json_file
  with open(output_json_path, 'w') as f:
    json.dump(video_caption.content, f)


def process_json_file(json_file, input_json_folder, output_json_folder, video_root, API):
  json_path = os.path.join(input_json_folder, json_file)
  update_json_path = os.path.join(output_json_folder, json_file)
  if os.path.exists(update_json_path):
    print(f"{update_json_path} is already exists")
    return 

  with open(json_path, 'r') as f:
      json_data = json.load(f)

  metadata = json_data['metadata']

  if metadata['cut'] == metadata['face_cut']:
      json_data['metadata']['face_cap'] = []
      with open(update_json_path, 'w') as f:
          json.dump(json_data, f, indent=4)
      return  

  crop_frames = extract_and_crop_frames(metadata, video_root)

  num_frames = 16
  temp_video_path = json_file.replace('.json','.mp4')
  video_base64 = save_frames_to_video_and_get_base64(crop_frames, temp_video_path, num_frames)

  video_caption = cap_video(video_base64, API)
  json_data['metadata']['face_cap'] = [video_caption.content]

  with open(update_json_path, 'w') as f:
    json.dump(json_data, f, indent=4)
    print(f"{update_json_path} has been saved")


def process_json_files_in_parallel(json_files, input_json_folder, output_json_folder, video_root, API, num_workers=4):
  with multiprocessing.Pool(num_workers) as pool:
    func = partial(process_json_file, input_json_folder=input_json_folder, 
                    output_json_folder=output_json_folder, video_root=video_root, API=API)
    
    pool.map(func, json_files)


def main():
  args = parse_args()

  #test
  json_test_dir = "/storage/hxy/ID/data/data_processor/sample/step4/cap_test/jsons"
  json_files = [f for f in os.listdir(json_test_dir) if f.endswith(".json")]
  output_json_dir = ""
  video_root = ""

  for json_file in json_files:
    process_json_file_test(json_file, json_test_dir, output_json_dir, video_root, args.API)

  # json_files = [f for f in os.listdir(args.input_json_folder) if f.endswith(".json")]

  # os.makedirs(args.output_json_folder, exist_ok=True)

  # process_json_files_in_parallel(json_files, args.input_json_folder, args.output_json_folder, args.video_root, args.API, num_workers=args.threads)


if __name__ == "__main__":
   main()