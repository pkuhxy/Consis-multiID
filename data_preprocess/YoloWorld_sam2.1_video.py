import os
import cv2
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import supervision as sv
from typing import Tuple
import pycocotools.mask as mask_util

from mmengine.config import Config, DictAction
from mmengine.runner.amp import autocast
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg

import torch
from torchvision.ops import nms
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import configs.transforms as T

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
MASK_ANNOTATOR = sv.MaskAnnotator()


class LabelAnnotator(sv.LabelAnnotator):

    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates,
        text_wh,
        position,
    ):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h


LABEL_ANNOTATOR = LabelAnnotator(text_padding=4,
                                 text_scale=0.5,
                                 text_thickness=1)


def estimate_num_people(data):
    def filter_by_confidence(objects, key, threshold=0.6):
        return [obj for obj in objects.get(key, []) if obj.get('confidence', 0) > threshold]

    max_people_PlanA = 0
    max_people_PlanB = 0
    max_people_PlanC = 0
    for frame_id, objects in data.items():
        filtered_faces = filter_by_confidence(objects, 'face')
        filtered_heads = filter_by_confidence(objects, 'head')
        filtered_persons = filter_by_confidence(objects, 'person')
        num_faces = len(filtered_faces)
        num_heads = len(filtered_heads)
        num_persons = len(filtered_persons)
        if num_persons == num_heads == num_faces:
            max_people_PlanA = max(max_people_PlanA, num_persons)
        if num_persons == num_heads:
            max_people_PlanB = max(max_people_PlanB, num_persons)
        if num_persons == num_faces:
            max_people_PlanC = max(max_people_PlanC, num_persons)
    if max_people_PlanA != 0:
        return max_people_PlanA
    elif max_people_PlanB != 0:
        return max_people_PlanB
    else:
        return max_people_PlanC


def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image_pil)
    image_transformed, _ = transform(image_pil, None)
    return image_np, image_transformed, image_pil


def inference_detector(model,
                       image_path,
                       texts,
                       test_pipeline,
                       max_dets=100,
                       score_thr=0.005,
                       nms_thr=0.5,
                       output_path='./work_dir',
                       use_amp=False,
                       annotation=False):
    data_info = dict(img_id=0, img_path=image_path, texts=texts)
    data_info = test_pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])

    with autocast(enabled=use_amp), torch.no_grad():
        output = model.test_step(data_batch)[0]
        pred_instances = output.pred_instances
        # pred_instances = pred_instances[pred_instances.scores.float() >
        #                                 score_thr]
    
    keep = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
    pred_instances = pred_instances[keep]
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

    if len(pred_instances.scores) > max_dets:
        indices = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[indices]

    pred_instances = pred_instances.cpu().numpy()

    bboxes = pred_instances['bboxes']
    class_ids = pred_instances['labels']
    confidence = pred_instances['scores']

    if annotation:
        if 'masks' in pred_instances:
            masks = pred_instances['masks']
        else:
            masks = None

        detections = sv.Detections(xyxy=bboxes,
                                class_id=class_ids,
                                confidence=confidence,
                                mask=masks)

        labels = [
            f"{texts[class_id][0]} {confidence:0.2f}" for class_id, confidence in
            zip(detections.class_id, detections.confidence)
        ]

        # label images
        image = cv2.imread(image_path)
        anno_image = image.copy()
        image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
        image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
        if masks is not None:
            image = MASK_ANNOTATOR.annotate(image, detections)
        cv2.imwrite(os.path.join(output_path, os.path.basename(image_path).replace(".png", "") + "_detection.png"), image)
    
    return bboxes, class_ids, confidence


def inference_all(
        args,
        yolo_model,
        test_pipeline,
        sam_model,
        texts,
        img_path,
        output_path
    ):
    """
    YoloWorld inference
    """
    # ##################################################################
    # image_np, image_transformed = load_image(img_path)
    # h, w, _ = image_np.shape

    # confidences = torch.tensor([0.6625, 0.6297, 0.4759, 0.4093, 0.4111])
    # input_boxes = np.array([[ 151.09076  ,  367.27615  ,  714.3426   , 1268.9829   ],
    #             [   1.5644073,  489.1618   ,  295.3487   , 1268.8892   ],
    #             [ 343.43475  ,  403.9524   ,  593.7483   ,  774.1255   ],
    #             [  81.35475  ,  480.40704  ,  245.61156  ,  594.95416  ],
    #             [  67.369675 ,  527.85675  ,  221.39104  ,  734.6142   ]], dtype=np.float32)
    # labels = ['person', 'person', 'person face', 'cap', 'person face']
    ##################################################################

    # reparameterize texts
    yolo_model.reparameterize(texts)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float32):
        bboxes, class_ids, confidence = inference_detector(
                                            model=yolo_model,
                                            image_path=img_path,
                                            texts=texts,
                                            test_pipeline=test_pipeline,
                                            max_dets=args.topk,
                                            score_thr=args.score_thr,
                                            nms_thr=args.nms_thr,
                                            output_path=output_path,
                                            use_amp=args.amp,
                                            annotation=args.bbox_annotation
                                        )


    """
    SAM inference
    """
    input_boxes = bboxes
    class_names = [texts[class_id][0] for class_id in class_ids]
    confidences = torch.from_numpy(confidence)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam_model.set_image(image_np)
        masks, scores, logits = sam_model.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
    
    return input_boxes, class_names, confidences, masks, scores


def post_process_all(
        img_path,
        input_boxes,
        class_names,
        confidences,
        scores,
        masks,
        output_path,
        dump_json_results,
    ):
    """
    Post-process the output of the model to get the masks, scores, and logits for visualization
    """
    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    class_ids = np.array(list(range(len(class_names))))

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]


    """
    Visualize image with supervision useful API
    """
    if args.all_annotation:
        img = cv2.imread(img_path)
        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.astype(bool),  # (n, h, w)
            class_id=class_ids
        )

        box_annotator = sv.BoundingBoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        cv2.imwrite(os.path.join(output_path, os.path.basename(img_path).replace(".png", "") + "_detection.png"), annotated_frame)

        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(os.path.join(output_path, os.path.basename(img_path).replace(".png", "") + "_segmentation.png"), annotated_frame)


    """
    Dump the results in standard format and save as json files
    """
    def single_mask_to_rle(mask):
        rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    if dump_json_results:
        # convert mask into rle format
        mask_rles = [single_mask_to_rle(mask) for mask in masks]

        input_boxes = input_boxes.tolist()
        scores = scores.tolist()
        # save the results in standard format
        results = {
            "image_path": os.path.basename(img_path),
            "annotations" : [
                {
                    "class_name": class_name,
                    "bbox": box,
                    "segmentation": mask_rle,
                    "score": score,
                }
                for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
            ],
            "box_format": "xyxy",
            "img_width": w,
            "img_height": h,
        }
        
        with open(os.path.join(output_path, os.path.basename(img_path).replace(".png", "").replace(".jpg", "") + ".json"), "w") as f:
            json.dump(results, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="YoloWorld-SAM2.1 Model Configuration")

    # all
    parser.add_argument('--text', default="person,cap,person face,person head", type=str, help='text prompts, including categories separated by a comma or a txt file with each line as a prompt.')
    parser.add_argument("--json_folder", default="step3/tags", help="Path to the folder containing JSON files.")
    parser.add_argument("--bbox_folder", default="step2/bbox", help="Path to the folder containing JSON files.")
    parser.add_argument('--video_folder', type=str, default="0_image/1.mp4", help="Path to the input image")
    parser.add_argument('--output_path', type=str, default="./demo_mask_outputs", help="Path to save the output")
    parser.add_argument('--dump_json_results', action="store_true", help="Flag to dump JSON results")
    parser.add_argument('--all_annotation', action='store_true', help='save the annotated detection results as YoloWorld-SAM2.1 format.')
    parser.add_argument('--bbox_annotation', action='store_true', help='save the annotated detection results as YOLO Text format.')
    
    # sam2.1
    parser.add_argument('--sam_checkpoint', type=str, default="/storage/ysh/Code/MultiID/Code/1_util_models/0_ckpts/sam2.1_hiera_large.pt", help="Path to the model sam_checkpoint")
    parser.add_argument('--sam_model_cfg', type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml", help="Path to the model configuration file")

    # yolo-world
    parser.add_argument('--yolo_config', default="configs/pretrain/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py", type=str, help='test config file path')
    parser.add_argument('--yolo_checkpoint', default="/storage/ysh/Code/MultiID/Code/1_util_models/ConsisID-X/YOLO-World-master/0_ckpts/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth", type=str, help='checkpoint file')
    parser.add_argument('--topk',
                        default=100,
                        type=int,
                        help='keep topk predictions.')
    parser.add_argument('--score_thr',
                        default=0.005,
                        type=float,
                        help='confidence score threshold for predictions.')
    parser.add_argument('--nms_thr',
                        default=0.5,
                        type=float,
                        help='confidence score threshold for predictions.')
    parser.add_argument('--device',
                        default='cuda',
                        help='device used for inference.')
    parser.add_argument('--amp',
                        action='store_true',
                        help='use mixed precision for inference.')
    parser.add_argument(
        '--cfg_options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    """
    Init input
    """
    sam_checkpoint = args.sam_checkpoint
    sam_model_cfg = args.sam_model_cfg
    video_folder = args.video_folder
    bbox_folder = args.bbox_folder
    output_path = args.output_path
    dump_json_results = args.dump_json_results

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    json_files = [os.path.join(args.json_folder, f) for f in os.listdir(args.json_folder) if f.endswith('.json')]


    """
    Load YoloWorld
    """
    cfg = Config.fromfile(args.yolo_config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.work_dir = os.path.join('./work_dirs',
                            os.path.splitext(os.path.basename(args.yolo_config))[0])
    # init model
    cfg.load_from = args.yolo_checkpoint
    yolo_model = init_detector(cfg, checkpoint=args.yolo_checkpoint, device=args.device)

    # init test pipeline
    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    # test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(test_pipeline_cfg)


    """
    Load SAM2.1
    """
    sam_model = SAM2ImagePredictor(build_sam2(sam_model_cfg, sam_checkpoint))


    """
    Main Loop
    """
    for json_file in tqdm(json_files, desc="Processing Files", total=len(json_files)):
        local_output_path = os.path.join(output_path, os.path.basename(json_file).replace(".json", ""))

        # get number of people by bbox annotation
        bbox_file = os.path.join(bbox_folder, os.path.basename(json_file))
        with open(bbox_file, "r") as f:
            bbox_data = json.load(f)
        max_num_people = estimate_num_people(bbox_data)

        # load input data
        with open(json_file, "r") as f:
            data = json.load(f)

        input_video_path = os.path.join(video_folder, data["video"])
        text_prompt = ". ".join(". ".join(tags).strip() for tags in data["word tags"].values() if tags).lower() + "."
        texts = [[tag.strip()] for tag in text_prompt.split('.') if tag.strip()] + [[' ']]
    
        # # load image
        # image_np, image_transformed, image_pil = load_image(img_path)
        # h, w, _ = image_np.shape
        # # load texts
        # texts = [[t.strip()] for t in args.text.split(',')] + [[' ']]  # [['person'], ['cap'], ['person face'], ['person head'], [' ']]


        # """
        # Main Inference
        # """
        # input_boxes, class_names, confidences, masks, scores = inference_all(
        #     args=args,
        #     yolo_model=yolo_model,
        #     test_pipeline=test_pipeline,
        #     sam_model=sam_model,
        #     texts=texts,
        #     img_path=img_path,
        #     output_path=output_path,
        # )


        # """
        # Post-process and Save
        # """
        # post_process_all(
        #     img_path=img_path,
        #     input_boxes=input_boxes,
        #     class_names=class_names,
        #     confidences=confidences,
        #     scores=scores,
        #     masks=masks,
        #     output_path=output_path,
        #     dump_json_results=dump_json_results,
        # )