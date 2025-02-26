# pip uninstall onnxruntime
# pip install onnxruntime-cann

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from diffusers.utils import load_image
from tqdm import tqdm

face_image = load_image("/work/share/projects/ysh/1.png")

# app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'AzureExecutionProvider', 'CANNExecutionProvider', 'CPUExecutionProvider'])
app = FaceAnalysis(name='antelopev2', root='./', providers=['CANNExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# prepare face emb
for _  in tqdm(range(100)):
    face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
    face_emb = face_info['embedding']