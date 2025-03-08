unset http_proxy
unset https_proxy

python step4_cap_videos.py --input_json_folder /storage/hxy/ID/data/data_processor/step2/step2_jsons/istockv4 \
    --output_json_folder /storage/hxy/ID/data/data_processor/step4/istockv4 \
    --video_root /storage/dataset/istock/ \
    --video_source istockv4 \
    --API 388291dd58fd4e1ab75b43664825a712.PzOLjKEDTQrKWDHP