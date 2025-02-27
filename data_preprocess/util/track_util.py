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


# from util.prepare_models import prepare_face_models

import json
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


def batch_cosine_similarity(embedding_image, embedding_frames, device="cuda"):
    embedding_image = torch.tensor(embedding_image).to(device)
    embedding_frames = torch.tensor(embedding_frames).to(device)
    return torch.nn.functional.cosine_similarity(embedding_image, embedding_frames, dim=-1).cpu().numpy()


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

    #将某一帧的某个人脸与标准embedding池比较，输出对应id(标准人脸池的id)
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

        return face_id

    def track_id(self):

        if self.standard_index == -1:
            return None

        standard_frame_id = {}
        for i in range(self.max_num):
            standard_frame_id[i] = i
        self.id_list[self.standard_index] = standard_frame_id

        #forward
        for index in range(self.standard_index, -1, -1):

            # import ipdb;ipdb.set_trace()

            self.id_list[index] = {}

            current_embeds = {}
            for id_index, face in enumerate(self.faces_infos[index]):
                face_id = self.get_id(face.embedding, self.embeds_forward)
                # 这里是指，第index帧，信息中的第id_index脸，对应人的id是face_id
                self.id_list[index][id_index] = face_id
                # 将标准池中的embbeding进行跟新
                self.embeds_forward[face_id] = face.embedding

            # self.embeds_forward = current_embeds

        #backward
        for index in range(self.standard_index, self.frames, 1):
            current_embeds = {}
            self.id_list[index] = {}

            # import ipdb;ipdb.set_trace()

            for id_index, face in enumerate(self.faces_infos[index]):
                face_id = self.get_id(face.embedding, self.embeds_backward)
                # 这里是指，第index帧，信息中的第id_index脸，对应人的id是face_id
                self.id_list[index][id_index] = face_id
                # 将标准池中的embbeding进行跟新
                self.embeds_backward[face_id] = face.embedding

            # self.embeds_backward = current_embeds


        return self.id_list


        