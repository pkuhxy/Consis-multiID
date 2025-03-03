import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
from moviepy import VideoFileClip
from tqdm import tqdm
import math


def parse_args():
    parser = argparse.ArgumentParser(description="Video Processing Parameters")
    parser.add_argument('--input_video_folder', type=str, default='step2/videos', help='Directory containing input videos (default: input_videos)')
    parser.add_argument('--input_json_folder', type=str, default='step2/bbox', help='Directory containing JSON files for bbox (default: step0/bbox)')
    parser.add_argument('--output_video_folder', type=str, default='step3/videos', help='Directory to store output videos (default: step1/videos)')
    parser.add_argument('--output_json_folder', type=str, default='step3/bbox', help='Directory to store output JSON files (default: step1/bbox)')
    parser.add_argument('--num_processes', type=int, default=1, help="Max number of parallel workers")
    args = parser.parse_args()
    return args


def filter_by_confidence(objects, key, threshold=0.6):
    return [obj for obj in objects.get(key, []) if obj.get('confidence', 0) > threshold]


def estimate_num_people(data):
    frame_result = {}
    max_people_PlanA = 0
    max_people_PlanB = 0
    max_people_PlanC = 0
    
    for frame_id, objects in data.items():
        frame_result[frame_id] = {
            # "face": 0,
            # "head": 0,
            # "person": 0,
            # "pose": 0,
            "num_people": 0
        }

        filtered_faces = filter_by_confidence(objects, 'face')
        filtered_heads = filter_by_confidence(objects, 'head')
        filtered_persons = filter_by_confidence(objects, 'person')
        filtered_poses = filter_by_confidence(objects, 'pose')

        num_faces = len(filtered_faces)
        num_heads = len(filtered_heads)
        num_persons = len(filtered_persons)
        num_pose = len(filtered_poses)

        # frame_result[frame_id]["face"] = num_faces
        # frame_result[frame_id]["head"] = num_heads
        # frame_result[frame_id]["person"] = num_persons
        # frame_result[frame_id]["pose"] = num_pose

        if num_persons == num_pose == num_heads or num_persons == num_pose == num_faces:
            frame_result[frame_id]["num_people"] = num_persons
        elif (num_persons == num_heads or num_persons == num_faces) and num_persons != num_pose:
            frame_result[frame_id]["num_people"] = num_persons
        elif (num_pose == num_heads or num_pose == num_faces) and num_persons != num_pose:
            frame_result[frame_id]["num_people"] = num_pose
        elif num_persons == num_pose:
            frame_result[frame_id]["num_people"] = num_persons
        elif num_persons == num_pose == num_heads == num_faces == 0:
            frame_result[frame_id]["num_people"] = 0
        else:
            frame_result[frame_id]["num_people"] = max(num_persons, num_pose, num_heads, num_faces)

        if num_persons == num_heads == num_faces:
            max_people_PlanA = max(max_people_PlanA, num_persons)
        if num_persons == num_heads:
            max_people_PlanB = max(max_people_PlanB, num_persons)
        if num_persons == num_faces:
            max_people_PlanC = max(max_people_PlanC, num_persons)

    if max_people_PlanA != 0:
        result = max_people_PlanA
    elif max_people_PlanB != 0:
        result = max_people_PlanB
    else:
        result = max_people_PlanC

    return result, frame_result


def is_same_people(frame_idx, frame_num_people_list, reference_num_people):
    return frame_num_people_list[str(frame_idx)]['num_people'] == reference_num_people

def extract_valid_segments_from_filtered_data(json_data, frame_num_people_list, min_length=16, tolerance=5):
    valid_segments = []
    current_segment = []
    consecutive_invalid_count = 0
    started_segment = False
    reference_num_people = None

    frame_keys = sorted(json_data.keys(), key=lambda x: int(x))

    # for frame_idx in frame_keys:
    i = 0
    while i < len(frame_keys):
        frame_idx = frame_keys[i]
        idx_int = int(frame_idx)

        if not started_segment:
            # Initialize a new segment
            started_segment = True
            reference_num_people = frame_num_people_list[frame_idx]['num_people']
            current_segment = [idx_int]
            consecutive_invalid_count = 0
        else:
            if is_same_people(frame_idx, frame_num_people_list, reference_num_people):
                # Valid frame, continue the segment
                current_segment.append(idx_int)
                consecutive_invalid_count = 0
            else:
                # Invalid frame
                consecutive_invalid_count += 1
                if consecutive_invalid_count <= tolerance:
                    # Add invalid frame within tolerance
                    current_segment.append(idx_int)
                else:
                    # Exceeded tolerance, finalize the current segment
                    trim_idx = len(current_segment) - 1
                    while trim_idx >= 0:
                        tail_key = str(current_segment[trim_idx])
                        if is_same_people(tail_key, frame_num_people_list, reference_num_people):
                            break
                        trim_idx -= 1

                    final_segment = current_segment[:trim_idx + 1]

                    if len(final_segment) >= min_length:
                        valid_segments.append(final_segment)

                    # started_segment = True
                    # reference_num_people = frame_num_people_list[frame_idx]['num_people']
                    # current_segment = [idx_int]
                    # consecutive_invalid_count = 0

                    # Restart segment and rollback to recheck frames
                    rollback_start = max(0, i - tolerance)
                    i = rollback_start - 1  # `-1` because `i` will be incremented in the next iteration

                    started_segment = False
                    current_segment = []
                    consecutive_invalid_count = 0
                    
        i += 1

    if len(current_segment) >= min_length:
        trim_idx = len(current_segment) - 1
        while trim_idx >= 0:
            tail_key = str(current_segment[trim_idx])
            if is_same_people(tail_key, frame_num_people_list, reference_num_people):
                break
            trim_idx -= 1
        final_segment = current_segment[:trim_idx + 1]
        if len(final_segment) >= min_length:
            valid_segments.append(final_segment)

    return valid_segments


def process_and_save_video(input_video_path, merged_segments, input_json_data, output_video_folder, output_json_folder):
    video_name = os.path.basename(input_video_path).replace('.mp4', '')
    video = VideoFileClip(input_video_path)

    segments_to_process = []
    for segment in merged_segments:
        start_frame = segment[0]
        end_frame = segment[-1]
        output_video_file = os.path.join(output_video_folder, f"{video_name}_{start_frame}_{end_frame}.mp4")
        output_bbox_file = os.path.join(output_json_folder, f"{video_name}_{start_frame}_{end_frame}.json")

        if not (os.path.exists(output_video_file) and os.path.exists(output_bbox_file)):
            segments_to_process.append(segment)

    if not segments_to_process:
        print("All segments already processed. Skipping video processing.")
        return

    for idx, segment in enumerate(segments_to_process):
        start_frame = segment[0]
        end_frame = segment[-1]
        start_time = start_frame / video.fps
        end_time = (end_frame + 1) / video.fps

        output_video_file = os.path.join(output_video_folder, f"{video_name}_{start_frame}_{end_frame}.mp4")
        output_bbox_file = os.path.join(output_json_folder, f"{video_name}_{start_frame}_{end_frame}.json")

        if not os.path.exists(output_video_file):
            video.subclipped(start_time, end_time).write_videofile(output_video_file, codec="libx264")

        if not os.path.exists(output_bbox_file):
            segment_json = {str(new_idx): input_json_data[str(original_idx)] for new_idx, original_idx in enumerate(segment)}
            with open(output_bbox_file, 'w') as f:
                json.dump(segment_json, f)

    print("Processing completed for the necessary segments.")


def process_video(input_video_path, input_json_path, output_video_folder, output_json_folder):
    video_name = os.path.basename(input_video_path).replace('.mp4', '')
    bbox_json_file = os.path.join(input_json_path, f"{video_name}.json")

    with open(bbox_json_file, 'r') as f:
        json_data = json.load(f)

    _, frame_num_people_list = estimate_num_people(json_data)

    useful_frame_list = extract_valid_segments_from_filtered_data(json_data, frame_num_people_list, tolerance=math.ceil(0.025*len(json_data)))

    process_and_save_video(input_video_path, useful_frame_list, json_data, output_video_folder, output_json_folder)


def main():
    args = parse_args()

    os.makedirs(args.output_video_folder, exist_ok=True)
    os.makedirs(args.output_json_folder, exist_ok=True)

    video_files = [f for f in os.listdir(args.input_video_folder) if f.endswith(".mp4")]

    # process_video(os.path.join(args.input_video_folder, "/remote-home1/ysh/1_Code/4_MultiID/1_MultiID/data_preprocess/step2/videos/test_0_257.mp4"), "/remote-home1/ysh/1_Code/4_MultiID/1_MultiID/data_preprocess/step2/json", args.output_video_folder, args.output_json_folder)
    with ThreadPoolExecutor(max_workers=args.num_processes) as executor:
        futures = [
            executor.submit(process_video, os.path.join(args.input_video_folder, video_file), args.input_json_folder, args.output_video_folder, args.output_json_folder)
            for video_file in video_files
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()


if __name__ == "__main__":
    main()
