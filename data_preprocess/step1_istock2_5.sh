ROOT=/storage/zhubin/meitu/istock
VIDEO_SOURCE=istock_v2
INPUT_VIDEO_JSON=/storage/lcm/ocr/final_istock_v2/istock_v2_final_254054.json
OUTPUT_JSON_FOLDER=/storage/hxy/ID/data/data_processor/step1_jsons
NUMS=576
MODEL_PATH=/storage/hxy/ID/ckpts/consisID

for i in {384..431}; do
    python step1_get_bbox_gpu.py --model_path ${MODEL_PATH} --part ${i} --device_id $((i % 4)) --video_source ${VIDEO_SOURCE} --video_root ${ROOT} --input_video_json ${INPUT_VIDEO_JSON} --output_json_folder ${OUTPUT_JSON_FOLDER} --split_nums ${NUMS} & \
done
wait


ROOT=/storage/dataset
VIDEO_SOURCE=sucai
INPUT_VIDEO_JSON=/storage/hxy/ID/data/dataset_check/filter_jsons/sucai_2778057.json
OUTPUT_JSON_FOLDER=/storage/hxy/ID/data/data_processor/step1_jsons
NUMS=576
MODEL_PATH=/storage/hxy/ID/ckpts/consisID

for i in {384..431}; do
    python step1_get_bbox_gpu.py --model_path ${MODEL_PATH} --part ${i} --device_id $((i % 4)) --video_source ${VIDEO_SOURCE} --video_root ${ROOT} --input_video_json ${INPUT_VIDEO_JSON} --output_json_folder ${OUTPUT_JSON_FOLDER} --split_nums ${NUMS} & \
done
wait