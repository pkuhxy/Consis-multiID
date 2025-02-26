CUDA_VISIBLE_DEVICES=0 python step1_get_bbox_kps.py --root /storage/dataset/panda70m \
    --video_source pandam9 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/pandam9_995234.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &


CUDA_VISIBLE_DEVICES=0 python step1_get_bbox_kps.py --root /storage/dataset/panda70m \
    --video_source pandam10 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/pandam10_1009613.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &


CUDA_VISIBLE_DEVICES=1 python step1_get_bbox_kps.py --root /storage/dataset/panda70m \
    --video_source pandam11 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/pandam11_1005070.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &

CUDA_VISIBLE_DEVICES=1 python step1_get_bbox_kps.py --root /storage/dataset/panda70m \
    --video_source pandam12 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/pandam12_1015012.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &

CUDA_VISIBLE_DEVICES=2 python step1_get_bbox_kps.py --root /storage/dataset/panda70m \
    --video_source pandam13 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/pandam13_1009222.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &

CUDA_VISIBLE_DEVICES=2 python step1_get_bbox_kps.py --root /storage/dataset/panda70m \
    --video_source pandam14 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/pandam14_1005729.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &

CUDA_VISIBLE_DEVICES=3 python step1_get_bbox_kps.py --root /storage/dataset/panda70m \
    --video_source pandam15 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/pandam15_1012827.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &

CUDA_VISIBLE_DEVICES=3 python step1_get_bbox_kps.py --root /storage/dataset/panda70m \
    --video_source pandam16 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/pandam16_1000543.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &

CUDA_VISIBLE_DEVICES=4 python step1_get_bbox_kps.py --root /storage/dataset/panda70m \
    --video_source pandam17 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/pandam17_665601.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &


CUDA_VISIBLE_DEVICES=4 python step1_get_bbox_kps.py --root /storage/dataset/panda70m \
    --video_source pandam18 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/pandam18_665285.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &


CUDA_VISIBLE_DEVICES=5 python step1_get_bbox_kps.py --root /storage/dataset/panda70m \
    --video_source pandam19 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/pandam19_681238.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &

CUDA_VISIBLE_DEVICES=5 python step1_get_bbox_kps.py --root /storage/dataset \
    --video_source vidal \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/vidal_1656690.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &

CUDA_VISIBLE_DEVICES=6 python step1_get_bbox_kps.py --root /storage/dataset/CTV \
    --video_source ctv01 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/ctv01_1473612.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &

CUDA_VISIBLE_DEVICES=6 python step1_get_bbox_kps.py --root /storage/dataset/CTV \
    --video_source ctv03 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/ctv03_698502.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &

CUDA_VISIBLE_DEVICES=7 python step1_get_bbox_kps.py --root /storage/dataset/CTV \
    --video_source ctv04 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/ctv04_688336.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &

CUDA_VISIBLE_DEVICES=7 python step1_get_bbox_kps.py --root /storage/dataset/CTV \
    --video_source ctv05 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/ctv05_305489.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 &