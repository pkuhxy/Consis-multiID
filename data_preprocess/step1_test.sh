ROOT=/storage/hxy/ID/data/data_processor/test
VIDEO_SOURCE=test_1
INPUT_VIDEO_JSON=/storage/hxy/ID/data/data_processor/test/istockv1_extracted.json
OUTPUT_JSON_FOLDER=/storage/hxy/ID/data/data_processor/test/step2_jsons
NUMS=8
MODEL_PATH=/storage/hxy/ID/ckpts/consisID

# python step1_get_bbox_gpu.py --model_path ${MODEL_PATH} --part 0 --device_id 0 --video_source ${VIDEO_SOURCE} --video_root ${ROOT} --input_video_json ${INPUT_VIDEO_JSON} --output_json_folder ${OUTPUT_JSON_FOLDER} --split_nums ${NUMS} 

for i in {0..7}; do
    python step1_get_bbox_gpu.py --model_path ${MODEL_PATH} --part ${i} --device_id $((i%8)) --video_source ${VIDEO_SOURCE} --video_root ${ROOT} --input_video_json ${INPUT_VIDEO_JSON} --output_json_folder ${OUTPUT_JSON_FOLDER} --split_nums ${NUMS} & \
done
wait