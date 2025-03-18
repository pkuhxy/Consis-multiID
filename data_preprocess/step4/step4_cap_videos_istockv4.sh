unset http_proxy
unset https_proxy

python step4_cap_api.py --input_json_folder /storage/hxy/ID/data/data_processor/step2/step2_jsons/istockv2 \
    --output_json_folder /storage/hxy/ID/data/data_processor/step4/istockv2 \
    --video_root /storage/zhubin/meitu/istock/ \
    --video_source istockv2 \
    --api_key da36c8ba202714f3c663577bf4c10c63.jfXshFLsTaHBavmz