# CUDA_VISIBLE_DEVICES=0 python step1_get_bbox_kps.py --root /storage/dataset \
#     --video_source sucai \
#     --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/sucai/istock_v4_chunk_1.json \
#     --output_json_folder /storage/hxy/ID/data/data_processor/track &



# CUDA_VISIBLE_DEVICES=1 python step1_get_bbox_kps.py --root /storage/dataset \
#     --video_source sucai \
#     --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/sucai/istock_v4_chunk_2.json \
#     --output_json_folder /storage/hxy/ID/data/data_processor/track &



# CUDA_VISIBLE_DEVICES=2 python step1_get_bbox_kps.py --root /storage/dataset \
#     --video_source sucai \
#     --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/sucai/istock_v4_chunk_3.json \
#     --output_json_folder /storage/hxy/ID/data/data_processor/track &



# CUDA_VISIBLE_DEVICES=3 python step1_get_bbox_kps.py --root /storage/dataset \
#     --video_source sucai \
#     --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/sucai/istock_v4_chunk_4.json \
#     --output_json_folder /storage/hxy/ID/data/data_processor/track &


# CUDA_VISIBLE_DEVICES=4 python step1_get_bbox_kps.py --root /storage/dataset \
#     --video_source sucai \
#     --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/sucai/istock_v4_chunk_5.json \
#     --output_json_folder /storage/hxy/ID/data/data_processor/track &



# CUDA_VISIBLE_DEVICES=5 python step1_get_bbox_kps.py --root /storage/dataset \
#     --video_source sucai \
#     --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/sucai/istock_v4_chunk_6.json \
#     --output_json_folder /storage/hxy/ID/data/data_processor/track &



# CUDA_VISIBLE_DEVICES=6 python step1_get_bbox_kps.py --root /storage/dataset \
#     --video_source sucai \
#     --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/sucai/istock_v4_chunk_7.json \
#     --output_json_folder /storage/hxy/ID/data/data_processor/track &



# CUDA_VISIBLE_DEVICES=7 python step1_get_bbox_kps.py --root /storage/dataset \
#     --video_source sucai \
#     --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/sucai/istock_v4_chunk_8.json \
#     --output_json_folder /storage/hxy/ID/data/data_processor/track &




ROOT=/storage/dataset/istock/
VIDEO_SOURCE=istock_v4
INPUT_VIDEO_JSON=/storage/hxy/ID/data/dataset_check/filter_jsons/istock_v4_984827.json
OUTPUT_JSON_FOLDER=/storage/hxy/ID/data/data_processor/step1_jsons
NUMS=768
MODEL_PATH=/storage/hxy/ID/ckpts/consisID

for i in {0..95}; do
    python step1_get_bbox_gpu.py --model_path ${MODEL_PATH} --part ${i} --device_id $((i % 8)) --video_source ${VIDEO_SOURCE} --video_root ${ROOT} --input_video_json ${INPUT_VIDEO_JSON} --output_json_folder ${OUTPUT_JSON_FOLDER} --split_nums ${NUMS} & \
done
wait