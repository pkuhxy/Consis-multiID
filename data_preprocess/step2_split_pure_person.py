import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


import cv2
import math
from moviepy import VideoFileClip
from tqdm import tqdm
import decord

def parse_args():
    parser = argparse.ArgumentParser(description="Video Processing Parameters")
    parser.add_argument('--input_video_folder', type=str, default='/storage/hxy/ID/data/data_processor/verification_jsons/2', help='Directory containing input videos (default: input_videos)')
    parser.add_argument('--input_json_folder', type=str, default='/storage/hxy/ID/data/data_processor/verification_jsons/2', help='Directory containing JSON files for bbox (default: step0/bbox)')
    parser.add_argument('--output_video_folder', type=str, default='/storage/hxy/ID/data/data_processor/test/step2_output/output_videos', help='Directory to store output videos (default: step1/videos)')
    parser.add_argument('--output_json_folder', type=str, default='/storage/hxy/ID/data/data_processor/sample/istockv4/step2_jsons', help='Directory to store output JSON files (default: step1/bbox)')
    parser.add_argument('--num_processes', type=int, default=16, help="Max number of parallel workers")
    args = parser.parse_args()
    return args


def is_face_large_enough_v2(face_boxes, threshold=0):
    for box in face_boxes:
        width = box['box']['x2'] - box['box']['x1']
        height = box['box']['y2'] - box['box']['y1']
        if width > threshold and height > threshold:
            return True
    return False


def extract_useful_frames(bbox_infos, min_valid_frames=81, tolerance=5):
    data = bbox_infos

    useful_frames = []
    current_segment = []
    non_face_count = 0

    for frame_num in range(len(data)):
        if str(frame_num) in data and data[str(frame_num)]['face']:
            face_boxes = data[str(frame_num)]['face']
            if is_face_large_enough_v2(face_boxes):
                current_segment.append(frame_num)
                non_face_count = 0
            else:
                if current_segment:
                    if non_face_count < tolerance:
                        current_segment.append(frame_num)
                        non_face_count += 1
                    else:
                        while non_face_count > 0:
                            if not is_face_large_enough_v2(data[str(current_segment[-1])]['face']):
                                current_segment.pop()
                                non_face_count -= 1
                            else:
                                break
                        if len(current_segment) >= min_valid_frames:
                            useful_frames.append(current_segment)
                        current_segment = []
                        non_face_count = 0
        else:
            if current_segment:
                if non_face_count < tolerance:
                    current_segment.append(frame_num)
                    non_face_count += 1
                else:
                    while non_face_count > 0:
                        if not is_face_large_enough_v2(data[str(current_segment[-1])]['face']):
                            current_segment.pop()
                            non_face_count -= 1
                        else:
                            break
                    if len(current_segment) >= min_valid_frames:
                        useful_frames.append(current_segment)
                    current_segment = []
                    non_face_count = 0

    if current_segment and len(current_segment) >= min_valid_frames:
        while non_face_count > 0:
            if not is_face_large_enough_v2(data[str(current_segment[-1])]['face']):
                current_segment.pop()
                non_face_count -= 1
            else:
                break
        if len(current_segment) >= min_valid_frames:
            useful_frames.append(current_segment)

    return useful_frames

def process_video(input_json_path, output_json_folder):

    json_name = os.path.basename(input_json_path)

    with open(input_json_path, 'r') as f:
        json_data = json.load(f)

    bbox_infos = json_data['bbox']
    meta_data = json_data['metadata']

    #Extract useful frames from bbox data
    useful_frames_bbox = extract_useful_frames(bbox_infos, tolerance=math.ceil(0.05*len(bbox_infos)))

    for segment in useful_frames_bbox:
        
        new_json_data = json_data.copy()
        new_json_data['metadata']['face_cut'] = [segment[0], segment[-1]+1]

        new_json_data['bbox'] = {str(i): bbox_infos[str(i)] for i in range(segment[0], segment[-1]+1)}
        
        output_json_name = json_name.replace('.json', f'_step2_{segment[0]}-{segment[-1]+1}.json')
        output_json_path = os.path.join(output_json_folder, output_json_name)


        with open(output_json_path, 'w') as f:
            json.dump(new_json_data, f, indent=4)





def main():
    args = parse_args()

    os.makedirs(args.output_video_folder, exist_ok=True)
    os.makedirs(args.output_json_folder, exist_ok=True)

    json_files = [f for f in os.listdir(args.input_json_folder) if f.endswith(".json")]

    input_json_folder = args.input_json_folder
    output_json_folder = args.output_json_folder


    for json_file in json_files:
        json_path = os.path.join(input_json_folder, json_file)
        process_video(json_path, output_json_folder)




if __name__ == "__main__":
    main()
