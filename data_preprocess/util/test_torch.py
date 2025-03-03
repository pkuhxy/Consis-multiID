# pip uninstall onnxruntime
# pip install onnxruntime-cann

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from diffusers.utils import load_image
from tqdm import tqdm
import torch

# import torch_npu
# from torch_npu.contrib import transfer_to_npu

face_image = load_image("/storage/hxy/ID/data/ID/test.png")

# app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'AzureExecutionProvider', 'CANNExecutionProvider', 'CPUExecutionProvider'])
app = FaceAnalysis(name='antelopev2', root='/storage/hxy/ID/ckpts/consisID/face_encoder', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def batch_cosine_similarity(embedding_image, embedding_frames, device="cuda"):
    embedding_image = torch.tensor(embedding_image).to(device)
    embedding_frames = torch.tensor(embedding_frames).to(device)
    return torch.nn.functional.cosine_similarity(embedding_image, embedding_frames, dim=-1).cpu().numpy()

# prepare face emb
for _  in tqdm(range(100)):
    face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))

    # import ipdb; ipdb.set_trace()
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
    face_emb = face_info['embedding']

    # batch_cosine_similarity(face_emb, face_emb)


