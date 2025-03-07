ROOT=/storage/zhubin/meitu/istock
VIDEO_SOURCE=istock_v1
INPUT_VIDEO_JSON=/storage/hxy/ID/data/dataset_check/filter_jsons/istock_v1_304948.json
OUTPUT_JSON_FOLDER=/storage/hxy/ID/data/data_processor/step1_jsons
NUMS=864
MODEL_PATH=/storage/hxy/ID/ckpts/consisID

for i in {672..767}; do
    python step1_get_bbox_gpu.py --model_path ${MODEL_PATH} --part ${i} --device_id $((i % 8)) --video_source ${VIDEO_SOURCE} --video_root ${ROOT} --input_video_json ${INPUT_VIDEO_JSON} --output_json_folder ${OUTPUT_JSON_FOLDER} --split_nums ${NUMS} & \
done
wait