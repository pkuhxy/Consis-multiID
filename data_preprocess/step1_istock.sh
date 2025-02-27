CUDA_VISIBLE_DEVICES=4 python step1_get_bbox_kps.py --root /storage/zhubin/meitu/istock \
    --video_source istock_v4 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/istock_v4/istock_v4_chunk_1.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1_track &



CUDA_VISIBLE_DEVICES=5 python step1_get_bbox_kps.py --root /storage/zhubin/meitu/istock \
    --video_source istock_v4 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/istock_v4/istock_v4_chunk_2.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1_track &



CUDA_VISIBLE_DEVICES=6 python step1_get_bbox_kps.py --root /storage/zhubin/meitu/istock \
    --video_source istock_v4 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/istock_v4/istock_v4_chunk_3.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1_track &



CUDA_VISIBLE_DEVICES=7 python step1_get_bbox_kps.py --root /storage/zhubin/meitu/istock \
    --video_source istock_v4 \
    --input_video_json /storage/hxy/ID/data/dataset_check/filter_jsons/istock_v4/istock_v4_chunk_4.json \
    --output_json_folder /storage/hxy/ID/data/data_processor/step1_track &