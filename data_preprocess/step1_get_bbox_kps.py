import argparse
import itertools
import json
import multiprocessing
import os
import torch
import cv2
import numpy as np
import time
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import decord
from insightface.app import FaceAnalysis
from functools import partial

from huggingface_hub import hf_hub_download
from tqdm import tqdm
from ultralytics import YOLO
from util.download_weights_data import download_file
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from util.track_util import IDTracker

# from util.prepare_models import prepare_face_models

import json
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image




def parse_args():
    parser = argparse.ArgumentParser(description="Process MP4 files with YOLO models.")
    parser.add_argument('--input_video_json', type=str, default='/storage/hxy/ID/data/dataset_check/scripts/istock_1000.json', help='Path to the folder containing MP4 files.')
    parser.add_argument('--output_json_folder', type=str, default='/storage/hxy/ID/data/data_processor/step1', help='Directory for output files.')
    parser.add_argument('--root', type=str, default='/storage/zhubin/meitu/istock', help='Directory for output files.')
    parser.add_argument('--video_source', type=str, default='test', help='Directory for output files.')
    parser.add_argument('--threads', type=int, default=12, help='Directory for output files.')

    return parser.parse_args()


def draw_rectangle_and_save(image, point1, point2, output_path):
    
    if image is None:
        print("Error: Could not load image.")
        return
    
    # 确保point1是左上角，point2是右下角
    x1, y1 = point1
    x2, y2 = point2
    top_left = (min(x1, x2), min(y1, y2))
    bottom_right = (max(x1, x2), max(y1, y2))
    
    # 在图像上绘制矩形框
    # 参数：图像、左上角坐标、右下角坐标、颜色（BGR格式）、线宽
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    
    # 保存绘制后的图像
    cv2.imwrite(output_path, image)
    print(f"Image with rectangle saved to {output_path}")


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


def pad_np_bgr_image(np_image, scale=1.25):
    assert scale >= 1.0, "scale should be >= 1.0"
    pad_scale = scale - 1.0
    h, w = np_image.shape[:2]
    top = bottom = int(h * pad_scale)
    left = right = int(w * pad_scale)
    ret = cv2.copyMakeBorder(np_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    return ret, (left, top)

def get_faces_info(face_helper, images, cut):

    faces_infos = {}
    origin_infos = {}
    images = images[cut[0]:cut[1]]

    detect_flag = False

    for index, image in enumerate(images):
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        faces_info = face_helper.get(image_bgr)

        # import ipdb; ipdb.set_trace()

        if len(faces_info) == 0:
            # padding, try again
            _h, _w = image_bgr.shape[:2]
            _img, left_top_coord = pad_np_bgr_image(image_bgr)
            faces_info = face_helper.get(_img)
            # if len(faces_info) == 0:
            #     print("Warning: No face detected in the image. Continue processing...")

            min_coord = np.array([0, 0])
            max_coord = np.array([_w, _h])
            sub_coord = np.array([left_top_coord[0], left_top_coord[1]])
            for face in faces_info:
                face.bbox = np.minimum(np.maximum(face.bbox.reshape(-1, 2) - sub_coord, min_coord), max_coord).reshape(4)
                face.kps = face.kps - sub_coord
        
        if len(faces_info) != 0:
            detect_flag = True

        origin_infos[index] = faces_info

        bboxs = []
        kpss_x = []
        kpss_y = []
        det_scores = []

        for face_info in faces_info:
            bboxs.append([int(x) for x in face_info.bbox.tolist()])
            kpss_x.append([int(x) for x in face_info.kps[:,0].tolist()])
            kpss_y.append([int(x) for x in face_info.kps[:,1].tolist()])
            det_scores.append(float(face_info.det_score))

        info_dict = {}

        info_dict['bboxs'] = bboxs
        info_dict['kpss_x'] = kpss_x
        info_dict['kpss_y'] = kpss_y
        info_dict['det_scores'] = det_scores

        # import ipdb;ipdb.set_trace()

        faces_infos[index] = info_dict

    
    if detect_flag:
        return faces_infos, origin_infos
    else:
        return None, None

def get_faces_info_RetinaFace(face_helper, images, cut):
    faces_infos = {}
    origin_infos = {}
    images = images[cut[0]:cut[1]]

    for index, image in enumerate(images):
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        faces_info = face_helper.face_det.detect_faces(image_bgr)

        if len(faces_info) == 0:
            # padding, try again
            _h, _w = image_bgr.shape[:2]
            _img, left_top_coord = pad_np_bgr_image(image_bgr)
            faces_info = face_helper.face_det.detect_faces(image_bgr)
            # if len(faces_info) == 0:
            #     print("Warning: No face detected in the image. Continue processing...")

            min_coord = np.array([0, 0])
            max_coord = np.array([_w, _h])
            sub_coord = np.array([left_top_coord[0], left_top_coord[1]])
            for face in faces_info:
                face.bbox = np.minimum(np.maximum(face.bbox.reshape(-1, 2) - sub_coord, min_coord), max_coord).reshape(4)
                face.kps = face.kps - sub_coord
        

        

        bboxs = []
        kpss_x = []
        kpss_y = []
        det_scores = []

        for face_info in faces_info:
            bboxs.append([int(x) for x in face_info[0:4].tolist()])
            det_scores.append(float(face_info[4]))
            kpss_x.append([int(x) for x in face_info[5::2].tolist()])
            kpss_y.append([int(x) for x in face_info[6::2].tolist()])

        info_dict = {}

        info_dict['bboxs'] = bboxs
        info_dict['kpss_x'] = kpss_x
        info_dict['kpss_y'] = kpss_y
        info_dict['det_scores'] = det_scores

        faces_infos[index] = info_dict

    return faces_infos, origin_infos
    


def check_id(faces_infos):

    id_list = {}
    embedding_map = {}
    max_index = 0
    max_faces = 0

    for key, value in faces_infos.items():
        if len(value) > max_faces :
            max_faces = len(value)
            max_index = key

    for index, face in enumerate(faces_infos[max_index]):
        embedding_map[index] = face.embedding

    for key, value in faces_infos.items():
        
        frame_id = {}

        for index, face in enumerate(value):
            max_score = -1
            max_id = -1
            for id_index, id_embedding in embedding_map.items():
                score = batch_cosine_similarity(face.embedding, id_embedding)
                if score > max_score:
                    max_score = score
                    max_id = id_index
            
            frame_id[index] = max_id

        id_list[key] = frame_id

    return id_list


def find_mp4_files(folder_path):
    mp4_files = []
    # 遍历文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.mp4'):  # 检查文件扩展名是否为 .mp4
                mp4_files.append(os.path.join(root, file))  # 获取绝对路径
    return mp4_files


def batch_cosine_similarity(embedding_image, embedding_frames, device="cuda"):
    embedding_image = torch.tensor(embedding_image).to(device)
    embedding_frames = torch.tensor(embedding_frames).to(device)
    return torch.nn.functional.cosine_similarity(embedding_image, embedding_frames, dim=-1).cpu().numpy()


def save_json(faces_info, output_path, id_list):

    json_data = {}

    for frame, face_info in faces_info.items():
        face = {}
        face['face'] = []

        bboxs = face_info['bboxs']
        kpss_x = face_info['kpss_x']
        kpss_y = face_info['kpss_y']
        det_scores = face_info['det_scores']
        face_num = len(bboxs)

        for index in range(face_num):
            box = {}
            box['x1'] = float(bboxs[index][0])
            box['y1'] = float(bboxs[index][1])
            box['x2'] = float(bboxs[index][2])
            box['y2'] = float(bboxs[index][3])
        
            kps_info = {}
            kps_info['x'] = kpss_x[index]
            kps_info['y'] = kpss_y[index]

            id_value = {}
            id_value['track_id'] = id_list[frame][index]
            id_value['box'] = box
            id_value['keypoints'] = kps_info
            id_value['confidence'] = float(det_scores[index])
            id_value['class'] = 0
            id_value['name'] = 'FACE'

            face['face'].append(id_value)


        json_data[frame] = face

    with open(output_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

        # import ipdb;ipdb.set_trace()

    return json_data

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):  # 如果是 NumPy 数组
        return obj.astype(float).tolist()  # 转换为 float 类型，再转换为列表
    elif isinstance(obj, (np.float32, np.float64)):  # 如果是 float32 或 float64
        return float(obj)  # 转换为 Python 的 float 类型
    elif isinstance(obj, (np.int32, np.int64)):  # 如果是 int32 或 int64
        return int(obj)  # 转换为 Python 的 int 类型
    elif isinstance(obj, dict):  # 如果是字典
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):  # 如果是列表
        return [convert_to_serializable(item) for item in obj]
    else:  # 其他类型（如字符串、整数等）保持不变
        return obj



def process_video(video_data, model_path, device, output_json_folder, video_source, root):
    # start_time = time.time()

    count = 0
    face_main_model = FaceAnalysis(
        name="antelopev2", root=os.path.join(model_path, "face_encoder"), providers=["CUDAExecutionProvider"]
    )
    face_main_model.prepare(ctx_id=0, det_size=(640, 640))

    face_helper_1 = FaceAnalysis(
        name="buffalo_l", root=os.path.join(model_path, "face_encoder"), providers=["CUDAExecutionProvider"]
    )
    face_helper_1.prepare(ctx_id=0, det_size=(640, 640))

    face_helper_2 = FaceAnalysis(
        name="buffalo_m", root=os.path.join(model_path, "face_encoder"), providers=["CUDAExecutionProvider"]
    )
    face_helper_2.prepare(ctx_id=0, det_size=(640, 640))

    face_helper_3 = FaceAnalysis(
        name="buffalo_s", root=os.path.join(model_path, "face_encoder"), providers=["CUDAExecutionProvider"]
    )
    face_helper_3.prepare(ctx_id=0, det_size=(640, 640))

    # face_helper_1 = FaceRestoreHelper(
    #     upscale_factor=1,
    #     face_size=512,
    #     crop_ratio=(1, 1),
    #     det_model="retinaface_resnet50",
    #     save_ext="png",
    #     device=device,
    #     model_rootpath=os.path.join(model_path, "face_encoder"),
    # )
    # face_helper_1.face_parse = None
    # face_helper_1.face_parse = init_parsing_model(
    #     model_name="bisenet", device=device, model_rootpath=os.path.join(model_path, "face_encoder")
    # )
    # face_helper_1.face_det.eval()
    # face_helper_1.face_parse.eval()
    # face_helper_1.face_det.to(device)
    # face_helper_1.face_parse.to(device)

    for video in video_data[0]:

        # import ipdb;ipdb.set_trace()

        count += 1


        cut = video['cut']
        path = os.path.join(root, video['path'])
        crop = video['crop']
        file_name = os.path.basename(path)
        json_name = file_name.replace('.mp4', '.json')

        output_dir = os.path.join(output_json_folder, video_source)

        # 检查文件是否已存在
        if os.path.exists(os.path.join(output_dir, json_name)):
            print(f"Skipping existing file: {json_name}")
            continue

        frames = extract_frames(path)
        if frames is None:
            continue

        faces_infos, origin_infos = get_faces_info(face_main_model, frames, cut)

        if faces_infos is None:
            faces_infos, origin_infos = get_faces_info(face_helper_1, frames, cut)

        if faces_infos is None:
            faces_infos, origin_infos = get_faces_info(face_helper_2, frames, cut)

        if faces_infos is None:
            faces_infos, origin_infos = get_faces_info(face_helper_3, frames, cut)

        if faces_infos is None:
            continue

        # import ipdb;ipdb.set_trace()

        Tracker = IDTracker(origin_infos)
        id_list = Tracker.track_id()


        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, json_name)

        save_json(faces_infos, output_path, id_list)
        print(f"process num : {count}, processing video: {video_source}")

        # end_time = time.time()  # 记录结束时间
        # elapsed_time = end_time - start_time  # 计算运行时间
        # print(f"main() 运行时间: {elapsed_time:.2f} 秒")


def test_tracker(model_path, test_dir, output_json_dir, output_video_dir):

    face_main_model = FaceAnalysis(
        name="antelopev2", root=os.path.join(model_path, "face_encoder"), providers=["CUDAExecutionProvider"]
    )
    face_main_model.prepare(ctx_id=0, det_size=(640, 640))

    face_helper_1 = FaceAnalysis(
        name="buffalo_l", root=os.path.join(model_path, "face_encoder"), providers=["CUDAExecutionProvider"]
    )
    face_helper_1.prepare(ctx_id=0, det_size=(640, 640))

    face_helper_2 = FaceAnalysis(
        name="buffalo_m", root=os.path.join(model_path, "face_encoder"), providers=["CUDAExecutionProvider"]
    )
    face_helper_2.prepare(ctx_id=0, det_size=(640, 640))

    face_helper_3 = FaceAnalysis(
        name="buffalo_s", root=os.path.join(model_path, "face_encoder"), providers=["CUDAExecutionProvider"]
    )

    face_helper_3.prepare(ctx_id=0, det_size=(640, 640))

    
    mp4_files = []
    for root, dirs, files in os.walk(test_dir):  # os.walk 会遍历文件夹及其子文件夹
        for f in files:
            if f.endswith('.mp4'):  # 判断文件后缀名是否为 .mp4
                mp4_files.append(os.path.join(root, f))  # 获取完整路径

    for video_file in mp4_files:
        frames = extract_frames(video_file)
        cut = [0, len(frames)]

        if frames is None:
            continue

        faces_infos, origin_infos = get_faces_info(face_main_model, frames, cut)

        if faces_infos is None:
            faces_infos, origin_infos = get_faces_info(face_helper_1, frames, cut)

        if faces_infos is None:
            faces_infos, origin_infos = get_faces_info(face_helper_2, frames, cut)

        if faces_infos is None:
            faces_infos, origin_infos = get_faces_info(face_helper_3, frames, cut)

        if faces_infos is None:
            continue

        file_name = os.path.basename(video_file)
        json_name = file_name.replace('.mp4', '.json')

        # import ipdb;ipdb.set_trace()

        Tracker = IDTracker(origin_infos)
        id_list = Tracker.track_id()
        if id_list == None:
            continue

        os.makedirs(output_json_dir, exist_ok=True)
        output_path = os.path.join(output_json_dir, json_name)



        json_data = save_json(faces_infos, output_path, id_list)

        output_path = os.path.join(output_video_dir, file_name)

        draw_bbox_and_track_id(video_file, json_data, output_path)

        
def draw_bbox_and_track_id(video_path, json_data, output_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频的基本信息
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建视频写入器
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # 遍历每一帧
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 获取当前帧的数据（假设 JSON 中的键是帧号）
        frame_data = json_data.get(frame_count)
        if frame_data:
            for face in frame_data["face"]:
                # 获取 bbox 和 track_id
                track_id = face["track_id"]
                box = face["box"]
                x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])

                # 绘制 bbox（矩形框）
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框

                # 在 bbox 上方添加 track_id 文本
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # import ipdb;ipdb.set_trace()

        

        # 写入处理后的帧到输出视频
        out.write(frame)

        # 增加帧计数
        frame_count += 1

    # 释放资源
    cap.release()
    out.release()
    print(f"处理完成，结果已保存到: {output_path}")

def split_data(data, num_processes):

    import ipdb;ipdb.set_trace()
    chunk_size = len(data) // num_processes
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

def main():
    start_time = time.time()
    args = parse_args()

    model_path = "/storage/hxy/ID/ckpts/consisID"
    device = "cuda"
    dtype = torch.bfloat16
    
    # 读取视频数据
    json_path = args.input_video_json
    root = args.root

    with open(json_path, "r") as f:
        data = json.load(f)

    # 切分数据
    num_processes = args.threads  # 根据机器性能选择适当的进程数
    video_chunks = split_data(data, num_processes)

    # test_dir = "/storage/hxy/ID/data/data_processor/verification"
    # output_json_dir = "/storage/hxy/ID/data/data_processor/verification_jsons"
    # output_video_dir = "/storage/hxy/ID/data/data_processor/verification_videos"

    # test_tracker(model_path, test_dir, output_json_dir, output_video_dir)
    # process_video(video_data=video_chunks, model_path=model_path, device=device, output_json_folder=args.output_json_folder, video_source=args.video_source, root=root)

    # 创建进程池，开始处理每个视频分块
    with multiprocessing.Pool(processes=num_processes) as pool:
        func = partial(process_video, model_path=model_path, device=device, output_json_folder=args.output_json_folder, video_source=args.video_source, root=root)
        pool.map(func, video_chunks)

    print("Processing completed in", time.time() - start_time, "seconds.")




    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间
    print(f"main() 运行时间: {elapsed_time:.2f} 秒")


if __name__ == "__main__":
    main()