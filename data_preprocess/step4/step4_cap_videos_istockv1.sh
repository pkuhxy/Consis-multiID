unset http_proxy
unset https_proxy

python step4_cap_api.py --input_json_folder /storage/hxy/ID/data/data_processor/step2/step2_jsons/sucai \
    --output_json_folder /storage/hxy/ID/data/data_processor/step4/sucai \
    --video_root /storage/dataset \
    --video_source sucai \
    --api_key da36c8ba202714f3c663577bf4c10c63.jfXshFLsTaHBavmz