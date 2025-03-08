import torch
from decord import VideoReader, cpu
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import importlib
if importlib.util.find_spec("torch_npu") is not None:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
else:
    torch_npu = None

import gc
import os
import cv2
import time
import json
import random
import argparse
import itertools
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
from insightface.app import FaceAnalysis
from multiprocessing import Pool

def free_memory():
    """
    Runs garbage collection. Then clears the cache of the available accelerator.
    """
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif is_torch_npu_available():
        torch_npu.npu.empty_cache()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


def pad_np_bgr_image(np_image, scale=1.25):
    assert scale >= 1.0, "scale should be >= 1.0"
    pad_scale = scale - 1.0
    h, w = np_image.shape[:2]
    top = bottom = int(h * pad_scale)
    left = right = int(w * pad_scale)
    ret = cv2.copyMakeBorder(np_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    return ret, (left, top)


def batch_cosine_similarity(embedding_image, embedding_frames, device="cuda"):
    embedding_image = torch.tensor(embedding_image).to(device)
    embedding_frames = torch.tensor(embedding_frames).to(device)
    return torch.nn.functional.cosine_similarity(embedding_image, embedding_frames, dim=-1).cpu().numpy()


def save_json(faces_info, output_path, id_list, frame_list, metadata):
    json_data = {
        "metadata": metadata,
        "bbox": {}
    }


    for i, (frame, face_info) in enumerate(faces_info.items()):
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

        json_data["bbox"][str(frame_list[i])] = face

    with open(output_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)
        print(f"Json file saved to {output_path}")

    return json_data


class IDTracker:
    def __init__(self, faces_infos):
        self.faces_infos = faces_infos
        self.id_list = {}

        self.embeds_forward = {}
        self.embeds_backward = {}

        self.standard_index = -1
        self.max_num = 0
        self.frames = len(faces_infos)
    
        max_index = []
        det_score = {}
        max_num = 0
        for index, faces_info in self.faces_infos.items():

            if len(faces_info) == 0:
                continue

            if len(faces_info) == max_num:
                max_index.append(index)
                
                score_sum = 0.0

                for face in faces_info:
                    score_sum += float(face.det_score)

                average_score = score_sum / len(faces_info)
                det_score[index] = average_score


            elif len(faces_info) > max_num:
                max_num = len(faces_info)
                max_index = []
                max_index.append(index)

                det_score = {}
                score_sum = 0.0
                for face in faces_info:
                    score_sum += float(face.det_score)

                average_score = score_sum / len(faces_info)
                det_score[index] = average_score

        self.max_num = max_num

        max_average_score = 0.0
        standard_index = -1
        for index in max_index:
            if det_score[index] > max_average_score:
                max_average_score = det_score[index]
                standard_index = index

        if len(max_index) >0 :

            for face_id, face_info in enumerate(self.faces_infos[standard_index]):
                self.embeds_forward[face_id] = face_info.embedding
                self.embeds_backward[face_id] = face_info.embedding

        self.standard_index = standard_index

    def get_id(self, embedding, standard_embed):

        if self.standard_index == -1:
            return None

        max_score = -1
        face_id = -1
        for id_index, stand_embedding in standard_embed.items():
            score = batch_cosine_similarity(stand_embedding, embedding)
            if score > max_score:
                max_score = score
                face_id = id_index

        if max_score < 0.1:
            return None

        return face_id

    def track_id(self):

        if self.standard_index == -1:
            return None

        standard_frame_id = {}
        for i in range(self.max_num):
            standard_frame_id[i] = i
        self.id_list[self.standard_index] = standard_frame_id

        for index in range(self.standard_index, -1, -1):

            self.id_list[index] = {}
            current_embeds = {}

            for id_index, face in enumerate(self.faces_infos[index]):
                if face.det_score > 0.65:
                    face_id = self.get_id(face.embedding, self.embeds_forward)
                    if face_id is None:
                        face_id = self.max_num + 1
                        self.max_num += 1
                        self.id_list[index][id_index] = face_id
                    else:
                        self.id_list[index][id_index] = face_id

                    current_embeds[face_id] = face.embedding
                else:
                    self.id_list[index][id_index] = -1
            
            for face_id, face in current_embeds.items():
                self.embeds_forward[face_id] = face

        for face_id, face in self.embeds_forward.items():
            if face_id in self.embeds_backward:
                continue
            else:
                self.embeds_backward[face_id] = face

        for index in range(self.standard_index, self.frames, 1):
            current_embeds = {}
            self.id_list[index] = {}

            for id_index, face in enumerate(self.faces_infos[index]):
                if face.det_score > 0.65:
                    face_id = self.get_id(face.embedding, self.embeds_backward)
                    if face_id is None:
                        face_id = self.max_num + 1
                        self.max_num += 1
                        self.id_list[index][id_index] = face_id
                    else:
                        self.id_list[index][id_index] = face_id

                    current_embeds[face_id] = face.embedding
                else:
                    self.id_list[index][id_index] = -1

            for face_id, face in current_embeds.items():
                self.embeds_backward[face_id] = face


        return self.id_list


def get_faces_info(face_helper, frames):
    faces_infos = {}
    origin_infos = {}

    detect_flag = False

    for index, image in enumerate(frames):
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        faces_info = face_helper.get(image_bgr)

        if len(faces_info) == 0:
            # padding, try again
            _h, _w = image_bgr.shape[:2]
            _img, left_top_coord = pad_np_bgr_image(image_bgr)
            faces_info = face_helper.get(_img)
            # if len(faces_info) == 0:
            #     print("Warning: No face detected in the video. Continue processing...")

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

        faces_infos[index] = info_dict
    
    if detect_flag:
        return faces_infos, origin_infos
    else:
        return None, None


def test_process_videos(model_path, input_video_dir, output_json_dir, device_id):
    face_main_model = FaceAnalysis(name="antelopev2", root=os.path.join(model_path, "face_encoder"), providers=["CUDAExecutionProvider"], provider_options=[{"device_id": device_id}])
    face_main_model.prepare(ctx_id=device_id, det_size=(640, 640))

    face_helper_1 = FaceAnalysis(name="buffalo_l", root=os.path.join(model_path, "face_encoder"), providers=["CUDAExecutionProvider"], provider_options=[{"device_id": device_id}])
    face_helper_1.prepare(ctx_id=device_id, det_size=(640, 640))

    face_helper_2 = FaceAnalysis(name="buffalo_m", root=os.path.join(model_path, "face_encoder"), providers=["CUDAExecutionProvider"], provider_options=[{"device_id": device_id}])
    face_helper_2.prepare(ctx_id=device_id, det_size=(640, 640))

    face_helper_3 = FaceAnalysis(name="buffalo_s", root=os.path.join(model_path, "face_encoder"), providers=["CUDAExecutionProvider"], provider_options=[{"device_id": device_id}])
    face_helper_3.prepare(ctx_id=device_id, det_size=(640, 640))

    free_memory()

    video_paths = [f for f in os.listdir(input_video_dir) if f.endswith(".mp4")]

    for video in tqdm(video_paths, desc="Processing videos", unit="video"):
        video_path = os.path.join(input_video_dir, video)

        try:
            vr = VideoReader(video_path)
            frame_list = np.arange(0, len(vr))
            frames = vr.get_batch(frame_list).asnumpy()        
            del vr
            free_memory()
        except Exception as e:
            print(e)
            continue

        if frames is None:
            continue

        faces_infos, origin_infos = get_faces_info(face_main_model, frames)

        if faces_infos is None:
            faces_infos, origin_infos = get_faces_info(face_helper_1, frames)

        if faces_infos is None:
            faces_infos, origin_infos = get_faces_info(face_helper_2, frames)

        if faces_infos is None:
            faces_infos, origin_infos = get_faces_info(face_helper_3, frames)

        if faces_infos is None:
            continue        

        Tracker = IDTracker(origin_infos)
        id_list = Tracker.track_id()

        metadata = []
        output_path = os.path.join(output_json_dir, video.replace(".mp4", ".json"))

        save_json(faces_infos, output_path, id_list, frame_list, metadata)        


def process_video(video_metadatas, model_path, output_json_folder, video_source, video_root, device_id):
    face_main_model = FaceAnalysis(name="antelopev2", root=os.path.join(model_path, "face_encoder"), providers=["CUDAExecutionProvider"], provider_options=[{"device_id": device_id}])
    face_main_model.prepare(ctx_id=device_id, det_size=(640, 640))

    face_helper_1 = FaceAnalysis(name="buffalo_l", root=os.path.join(model_path, "face_encoder"), providers=["CUDAExecutionProvider"], provider_options=[{"device_id": device_id}])
    face_helper_1.prepare(ctx_id=device_id, det_size=(640, 640))

    face_helper_2 = FaceAnalysis(name="buffalo_m", root=os.path.join(model_path, "face_encoder"), providers=["CUDAExecutionProvider"], provider_options=[{"device_id": device_id}])
    face_helper_2.prepare(ctx_id=device_id, det_size=(640, 640))

    face_helper_3 = FaceAnalysis(name="buffalo_s", root=os.path.join(model_path, "face_encoder"), providers=["CUDAExecutionProvider"], provider_options=[{"device_id": device_id}])
    face_helper_3.prepare(ctx_id=device_id, det_size=(640, 640))

    free_memory()

    output_paths = []

    for metadata in tqdm(video_metadatas, desc="Processing videos", unit="video"):
        cut = metadata['cut']
        crop = metadata['crop']

        video_path = os.path.join(video_root, metadata['path'])
        output_dir = os.path.join(output_json_folder, video_source)
        output_name = os.path.basename(metadata['path']).replace(".mp4", f"_step1-{cut[0]}-{cut[1]}.json")
        output_path = os.path.join(output_dir, output_name)
        output_paths.append(output_path)

        os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(output_path):
            print(f"Skipping existing file: {output_name}")
            continue
        try:
            vr = VideoReader(video_path)
            frame_list = np.arange(cut[0], cut[1])
            frames = vr.get_batch(frame_list).asnumpy()
            del vr
            free_memory()
        except Exception as e:
            print(e)
            continue

        if frames is None:
            continue

        faces_infos, origin_infos = get_faces_info(face_main_model, frames)

        if faces_infos is None:
            faces_infos, origin_infos = get_faces_info(face_helper_1, frames)

        if faces_infos is None:
            faces_infos, origin_infos = get_faces_info(face_helper_2, frames)

        if faces_infos is None:
            faces_infos, origin_infos = get_faces_info(face_helper_3, frames)

        if faces_infos is None:
            flag = 'None'
            with open(output_path, 'w') as json_file:
                json.dump(flag, json_file, indent=4)
            print(f"Json file saved to {output_path}")
            continue

        Tracker = IDTracker(origin_infos)
        id_list = Tracker.track_id()

        save_json(faces_infos, output_path, id_list, frame_list, metadata)

        free_memory()

    save_jsons_path = f"/storage/hxy/ID/data/data_processor/test/check_save_jsons/save_jsons/{device_id}.json"
    with open(save_jsons_path, "w") as f:
        json.dump(output_paths, f, indent=4)


def split_list(data, nums, part):
    size = len(data)
    part_size = size // nums
    remainder = size % nums  # 剩余部分

    start = part * part_size + min(part, remainder)
    end = start + part_size + (1 if part < remainder else 0)

    index = [start,end]
    index_json_path = f"/storage/hxy/ID/data/data_processor/test/check_save_jsons/index_jsons/{part}.json"
    with open(index_json_path, "w") as f:
        json.dump(index, f, indent=4)

    return data[start:end]


def process_metadata(metadata, output_dir):
    cut = metadata['cut']
    output_name = os.path.basename(metadata['path']).replace(".mp4", f"_step1-{cut[0]}-{cut[1]}.json")
    output_path = os.path.join(output_dir, output_name)

    if os.path.exists(output_path):
        print(f"Skipping existing file: {output_name}")
        return None 

    return metadata 


def resume_data(data, output_dir, num_processes=4):
    with Pool(processes=num_processes) as pool:
        result = pool.starmap(process_metadata, [(metadata, output_dir) for metadata in data])

    resume_data = [metadata for metadata in result if metadata is not None]

    return resume_data


def parse_args():
    parser = argparse.ArgumentParser(description="Process MP4 files with YOLO models.")
    parser.add_argument('--model_path', type=str, default='/storage/hxy/ID/ckpts/consisID')
    parser.add_argument('--device_id', type=int, default=3)
    parser.add_argument('--input_video_json', type=str, default='/storage/hxy/ID/data/data_processor/test/istockv1_extracted.json', help='Path to the folder containing MP4 files.')
    parser.add_argument('--output_json_folder', type=str, default='/storage/hxy/ID/data/data_processor/test/wrong_examples/frames_gm621059908', help='Directory for output files.')
    parser.add_argument('--video_root', type=str, default='/storage/hxy/ID/data/data_processor/test', help='Directory for output files.')
    parser.add_argument('--video_source', type=str, default='sucai', help='Directory for output files.')
    parser.add_argument('--part', type=int, default=0, help='Directory for output files.')
    parser.add_argument('--split_nums', type=int, default=8, help='Directory for output files.')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    model_path = args.model_path
    json_path = args.input_video_json
    video_root = args.video_root
    output_dir = args.output_json_folder + '/' + args.video_source

    with open(json_path, "r") as f:
        data = json.load(f)

    data = resume_data(data, output_dir, 32)

    split_data = split_list(data, args.split_nums, args.part)

    start_time = time.time()
    process_video(video_metadatas=split_data, model_path=model_path, output_json_folder=args.output_json_folder, video_source=args.video_source, video_root=video_root, device_id=args.device_id)
    print("Processing completed in", time.time() - start_time, "seconds.")