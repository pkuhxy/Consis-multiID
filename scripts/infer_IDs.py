import torch
from diffusers import ConsisIDPipeline
from diffusers.pipelines.consisid.consisid_utils import prepare_face_models, process_face_embeddings_infer
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download
import pandas as pd

face_helper_1, face_helper_2, face_clip_model, face_main_model, eva_transform_mean, eva_transform_std = (
    prepare_face_models("/storage/hxy/ID/ckpts/consisID", device="cuda", dtype=torch.bfloat16)
    )

pipe = ConsisIDPipeline.from_pretrained("/storage/hxy/ID/ckpts/consisID", torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "The video features a young man who appears to be a content creator or streamer. he is wearing a green sleeveless top and red headphones. The background is illuminated with vibrant neon lights, predominantly in shades of purple and blue, creating a lively and energetic atmosphere. The man is seated in front of a microphone, suggesting he is recording a podcast, streaming a live broadcast, or engaging in some form of online communication. The setting appears to be a well-lit room with a curtain and a lamp visible in the background, adding to the cozy and inviting ambiance. The man's expression and body language indicate he is actively speaking or singing into the microphone, possibly sharing his thoughts, stories, or performing music. The overall scene conveys a sense of engagement and interaction, likely aimed at an audience who is tuned in to his content."

image = "/storage/hxy/ID/data/ID/face.jpg"

save_path = "/storage/hxy/ID/data/infer/consisID/hxy.mp4"


def infer_video(image, prompt, save_path):

    id_cond, id_vit_hidden, image, face_kps = process_face_embeddings_infer(
        face_helper_1,
        face_clip_model,
        face_helper_2,
        eva_transform_mean,
        eva_transform_std,
        face_main_model,
        "cuda",
        torch.bfloat16,
        image,
        is_align_face=True,
    )

    video = pipe(
        image=image,
        prompt=prompt,
        num_inference_steps=50,
        guidance_scale=6.0,
        use_dynamic_cfg=False,
        id_vit_hidden=id_vit_hidden,
        id_cond=id_cond,
        kps_cond=face_kps,
        generator=torch.Generator("cuda").manual_seed(42),
    )

    export_to_video(video.frames[0], save_path, fps=8)


prompts_files = "/storage/hxy/ID/data/ID/prompt.xlsx"
img_dir = "/storage/hxy/ID/data/ID"
save_dir = "/storage/hxy/ID/data/infer/consisID"


infer_video(image, prompt, save_path)

# df = pd.read_excel(prompts_files, header=None)
# prompts = df[1].tolist()

# for i in range(len(prompts)):
#     image = img_dir + '/' + str(i) + '.png'
#     prompt = prompts[i]
#     save_path = save_dir + '/' + str(i) + '.mp4'

#     # import ipdb; ipdb.set_trace()

#     infer_video(image, prompt, save_path)




