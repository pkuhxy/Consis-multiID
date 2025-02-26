CUDA_VISIBLE_DEVICES=0 python step1_get_bbox_kps.py --root /storage/zhubin/meitu/istock \
    --video_source istock_v1 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/istock_v1_304948.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &


CUDA_VISIBLE_DEVICES=0 python step1_get_bbox_kps.py --root /storage/zhubin/meitu/istock \
    --video_source istock_v4 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/istock_v4_984827.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &


CUDA_VISIBLE_DEVICES=1 python step1_get_bbox_kps.py --root /storage/dataset \
    --video_source sucai \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/sucai_2778057.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &

CUDA_VISIBLE_DEVICES=1 python step1_get_bbox_kps.py --root /storage/dataset/movie \
    --video_source bbc01 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/bbc01_222604.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &

CUDA_VISIBLE_DEVICES=2 python step1_get_bbox_kps.py --root /storage/dataset/movie \
    --video_source bbc02 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/bbc02_268121.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &

CUDA_VISIBLE_DEVICES=2 python step1_get_bbox_kps.py --root /storage/dataset/movie \
    --video_source bbc03 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/bbc03_954670.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &

CUDA_VISIBLE_DEVICES=3 python step1_get_bbox_kps.py --root /storage/dataset/movie \
    --video_source bbc04 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/bbc04_356557.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &

CUDA_VISIBLE_DEVICES=3 python step1_get_bbox_kps.py --root /storage/dataset/movie \
    --video_source bbc05 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/bbc05_592518.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &

CUDA_VISIBLE_DEVICES=4 python step1_get_bbox_kps.py --root /storage/dataset/panda70m \
    --video_source pandam1 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/pandam1_974058.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &


CUDA_VISIBLE_DEVICES=4 python step1_get_bbox_kps.py --root /storage/dataset/panda70m \
    --video_source pandam2 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/pandam2_1009139.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &


CUDA_VISIBLE_DEVICES=5 python step1_get_bbox_kps.py --root /storage/dataset/panda70m \
    --video_source pandam3 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/pandam3_935778.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &

CUDA_VISIBLE_DEVICES=5 python step1_get_bbox_kps.py --root /storage/dataset/panda70m \
    --video_source pandam4 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/pandam4_1004642.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &

CUDA_VISIBLE_DEVICES=6 python step1_get_bbox_kps.py --root /storage/dataset/panda70m \
    --video_source pandam5 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/pandam5_1008295.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &

CUDA_VISIBLE_DEVICES=6 python step1_get_bbox_kps.py --root /storage/dataset/panda70m \
    --video_source pandam6 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/pandam6_996096.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &

CUDA_VISIBLE_DEVICES=7 python step1_get_bbox_kps.py --root /storage/dataset/panda70m \
    --video_source pandam7 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/pandam7_1015174.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &

CUDA_VISIBLE_DEVICES=7 python step1_get_bbox_kps.py --root /storage/dataset/panda70m \
    --video_source pandam8 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/pandam8_998957.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &