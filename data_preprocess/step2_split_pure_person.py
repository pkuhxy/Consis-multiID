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
    parser.add_argument('--input_video_folder', type=str, default='/storage/hxy/ID/data/data_processor/test/step2_test_videos', help='Directory containing input videos (default: input_videos)')
    parser.add_argument('--input_json_folder', type=str, default='/storage/hxy/ID/data/data_processor/test/step2_test', help='Directory containing JSON files for bbox (default: step0/bbox)')
    parser.add_argument('--output_video_folder', type=str, default='/storage/hxy/ID/data/data_processor/test/step2_output/output_videos', help='Directory to store output videos (default: step1/videos)')
    parser.add_argument('--output_json_folder', type=str, default='/storage/hxy/ID/data/data_processor/test/step2_output/bbox_jsons', help='Directory to store output JSON files (default: step1/bbox)')
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


def extract_useful_frames(bbox_infos, video_file_path, min_valid_frames=81, tolerance=5):
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


def extract_frames(video_path):
    # 打开视频文件
    try:
        cap = decord.VideoReader(video_path)
        frames = []

        # 获取视频的总帧数
        frame_count = len(cap)

        if frame_count == 0:
            print("Error: Video file is empty or cannot be read.")
            return
        
        for frame_idx in range(frame_count):
            # 读取指定索引的帧
            frame = cap[frame_idx].asnumpy()  # 转换为 NumPy 数组
            
            frames.append(frame)

        print(f"Total frames: {frame_count}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

    return frames

def is_valid_frame(frame_data):
    for person in frame_data:
        visible = person['keypoints']['visible']
        # is_full_face = all(visible[i] >= 0.6 for i in range(3))  # 0: Nose, 1: Left Eye, 2: Right Eye
        # if is_full_face:
        #     return True
        is_half_face = (
            visible[0] >= 0.6 and 
            (visible[1] >= 0.6 or visible[2] >= 0.6)  # Nose and at least one eye visible
        )
        if is_half_face:
            return True
    return False


def merge_segments(useful_frames_bbox, segments_pose):
    merged_segments = []
    for bbox_segment in useful_frames_bbox:
        for pose_segment in segments_pose:
            # Find overlap between bbox and pose segments
            overlap = set(bbox_segment) & set(pose_segment)
            if overlap:
                # Merge the segment if there is overlap
                merged_segment = sorted(overlap)
                merged_segments.append(merged_segment)
    return merged_segments


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

        # if not os.path.exists(output_bbox_file):
        #     segment_json = {str(new_idx): input_json_data[str(original_idx)] for new_idx, original_idx in enumerate(segment)}
        #     with open(output_bbox_file, 'w') as f:
        #         json.dump(segment_json, f)

    print("Processing completed for the necessary segments.")


def extract_valid_segments_from_filtered_data(filtered_pose_json_data, min_valid_frames=81, tolerance=5):
    valid_segments = []
    current_segment = []
    consecutive_invalid_count = 0
    started_segment = False

    # for frame_idx, frame_data in filtered_pose_json_data.items():
    frame_keys = sorted(filtered_pose_json_data.keys(), key=lambda x: int(x))
    
    i = 0
    while i < len(frame_keys):
        frame_idx = frame_keys[i]
        frame_data = filtered_pose_json_data[frame_idx]

        if is_valid_frame(frame_data):
            if not started_segment:
                started_segment = True
            current_segment.append(int(frame_idx))
            consecutive_invalid_count = 0
        else:
            if not started_segment:
                i += 1
                continue

            consecutive_invalid_count += 1
            if consecutive_invalid_count <= tolerance:
                current_segment.append(int(frame_idx))
            else:
                trim_idx = len(current_segment) - 1
                while trim_idx >= 0:
                    tail_key = str(current_segment[trim_idx])
                    if is_valid_frame(filtered_pose_json_data[tail_key]):
                        break
                    trim_idx -= 1

                final_segment = current_segment[:trim_idx+1]
                if len(final_segment) >= min_valid_frames:
                    valid_segments.append(final_segment)

                # current_segment = []
                # consecutive_invalid_count = 0
                # started_segment = False
                
                # Restart segment and rollback to recheck frames
                rollback_start = max(0, i - tolerance)
                i = rollback_start - 1  # `-1` because `i` will be incremented in the next iteration

                current_segment = []
                consecutive_invalid_count = 0
                started_segment = False

        i += 1

    if len(current_segment) >= min_valid_frames:
        trim_idx = len(current_segment) - 1
        while trim_idx >= 0:
            tail_key = str(current_segment[trim_idx])
            if is_valid_frame(filtered_pose_json_data[tail_key]):
                break
            trim_idx -= 1

        final_segment = current_segment[:trim_idx+1]
        if len(final_segment) >= min_valid_frames:
            valid_segments.append(final_segment)

    return valid_segments

def save_frames_to_video(video_frames, start_frame, end_frame, output_video_file, fps=30):
    """
    将 start_frame 到 end_frame 之间的帧保存为视频。
    
    :param video_frames: 提取的所有视频帧 (列表，每个元素是一个帧图像)
    :param start_frame: 起始帧索引
    :param end_frame: 结束帧索引
    :param output_video_file: 输出视频文件路径
    :param fps: 输出视频的帧率，默认为 30
    """
    # 获取帧的尺寸
    height, width, _ = video_frames[0].shape

    # 定义视频编码器并创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 编码格式
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    # 遍历指定帧范围并写入视频
    for i in range(start_frame, end_frame + 1):
        out.write(video_frames[i])

    out.release()  # 释放资源
    print(f"视频已保存至: {output_video_file}")


def process_video(input_video_path, input_json_path, output_video_folder, output_json_folder):
    video_name = os.path.basename(input_video_path).replace('.mp4', '')
    video_json_file = os.path.join(input_json_path, f"{video_name}.json")
    useful_frames_json_file = os.path.join(output_json_folder, f"{video_name}_frames.json")

    with open(video_json_file, 'r') as f:
        json_data = json.load(f)

    bbox_infos = json_data['bbox']
    meta_data = json_data['metadata']


    # Step 1: Extract useful frames from bbox data
    useful_frames_bbox = extract_useful_frames(bbox_infos, input_video_path, tolerance=math.ceil(0.05*len(json_data)))

    new_infos = []

    for segment in useful_frames_bbox:
        new_info = meta_data.copy()
        meta_data['cut'] = segment
        new_infos.append(new_info)

    return new_infos

    # video_frames = extract_frames(input_video_path)
    # video_frames_bgr = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in video_frames]

    # for segment in useful_frames_bbox:
    #     start_frame = segment[0]
    #     end_frame = segment[-1]
    #     output_video_file = os.path.join(output_video_folder, f"{video_name}_{start_frame}_{end_frame}.mp4")
    #     save_frames_to_video(video_frames_bgr, start_frame, end_frame, output_video_file)




def main():
    args = parse_args()

    os.makedirs(args.output_video_folder, exist_ok=True)
    os.makedirs(args.output_json_folder, exist_ok=True)

    video_files = [f for f in os.listdir(args.input_video_folder) if f.endswith(".mp4")]

    # input_video_path = '/storage/hxy/ID/data/data_processor/verification/3.mp4'
    input_json_folder = args.input_json_folder
    output_json_folder = args.output_json_folder


    for video in video_files:

        video = os.path.join(args.input_video_folder, video)
        process_video(video, input_json_folder, args.output_video_folder, output_json_folder)



    # with ThreadPoolExecutor(max_workers=args.num_processes) as executor:
    #     futures = [
    #         executor.submit(process_video, os.path.join(args.input_video_folder, video_file), args.input_json_folder, args.output_video_folder, args.output_json_folder)
    #         for video_file in video_files
    #     ]
    #     for future in tqdm(as_completed(futures), total=len(futures)):
    #         future.result()


if __name__ == "__main__":
    main()
