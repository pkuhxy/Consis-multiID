INPUT_JSON=/storage/hxy/ID/data/data_processor/step4/merge_metadata_jsons/sucai_metadata.json
VIDEO_ROOT=/storage/dataset
SAVE_ROOT=/storage/hxy/ID/data/data_processor/step4/Aria/sucai
TORAL_PART=8



python ../step4_cap_Aria_transformer.py --video_root ${VIDEO_ROOT} --input_json ${INPUT_JSON} --save_root ${SAVE_ROOT} --part 1 --total_part 8 --shuffle