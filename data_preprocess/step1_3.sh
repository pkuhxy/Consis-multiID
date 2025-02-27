CUDA_VISIBLE_DEVICES=2 python step1_get_bbox_kps.py --root /storage/dataset/xigua_sourcecode \
    --video_source tiyu \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/tiyu_382538.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 \
    --threads 12 &



CUDA_VISIBLE_DEVICES=4 python step1_get_bbox_kps.py --root /storage/dataset/xigua_sourcecode \
    --video_source xigua_v1 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/xigua_v1_1537971.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 \
    --threads 12 &


CUDA_VISIBLE_DEVICES=5 python step1_get_bbox_kps.py --root /storage/dataset/xigua_sourcecode \
    --video_source xigua_v2 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/xigua_v2_3851705.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 \
    --threads 12 &



CUDA_VISIBLE_DEVICES=6 python step1_get_bbox_kps.py --root /storage/dataset/xigua_sourcecode \
    --video_source xigua_v3 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/xigua_v3_5348637.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 \
    --threads 12 &


CUDA_VISIBLE_DEVICES=7 python step1_get_bbox_kps.py --root /storage/dataset/xigua_sourcecode \
    --video_source xigua_v4 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/xigua_v4_1875289.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1 \
    --threads 12 &




# CUDA_VISIBLE_DEVICES=5 python step1_get_bbox_kps.py --root /storage/dataset/panda70m \
#     --video_source ctv \
#     --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/ctv_856960.json \
#     --output_json_folder /storage/hxy/ID/data/data_processor/step1 &

# CUDA_VISIBLE_DEVICES=5 python step1_get_bbox_kps.py --root /storage/dataset \
#     --video_source vidal \
#     --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/vidal_1656690.json \
#     --output_json_folder /storage/hxy/ID/data/data_processor/step1 &

# CUDA_VISIBLE_DEVICES=6 python step1_get_bbox_kps.py --root /storage/dataset/CTV \
#     --video_source ctv01 \
#     --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/ctv01_1473612.json \
#     --output_json_folder /storage/hxy/ID/data/data_processor/step1 &

# CUDA_VISIBLE_DEVICES=6 python step1_get_bbox_kps.py --root /storage/dataset/CTV \
#     --video_source ctv03 \
#     --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/ctv03_698502.json \
#     --output_json_folder /storage/hxy/ID/data/data_processor/step1 &

# CUDA_VISIBLE_DEVICES=7 python step1_get_bbox_kps.py --root /storage/dataset/CTV \
#     --video_source ctv04 \
#     --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/ctv04_688336.json \
#     --output_json_folder /storage/hxy/ID/data/data_processor/step1 &

# CUDA_VISIBLE_DEVICES=7 python step1_get_bbox_kps.py --root /storage/dataset/CTV \
#     --video_source ctv05 \
#     --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/ctv05_305489.json \
#     --output_json_folder /storage/hxy/ID/data/data_processor/step1 &