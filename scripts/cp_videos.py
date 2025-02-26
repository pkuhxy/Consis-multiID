import json
import os
import random
import ffmpeg
from concurrent.futures import ProcessPoolExecutor, as_completed

# 读取json文件
with open('/storage/lcm/ocr/final_istock_v1/istock_v1_final_321725.json', 'r') as f:
    data = json.load(f)

sample_size = 1000
if len(data) > sample_size:
    selected_data = random.sample(data, sample_size)
else:
    selected_data = data

with open('/storage/hxy/ID/Consis-multiID/jsons/selected_data.json', 'w') as f:
    json.dump(selected_data, f, ensure_ascii=False, indent=4)

# 指定目标文件夹
destination_folder = '/storage/hxy/ID/data/test_source_videos'
root_dir = '/storage/zhubin/meitu/istock'

# 确保目标文件夹存在，如果不存在则创建
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

def process_video(item, root_dir, destination_folder):
    video_path = item.get('path')  # 获取path对应的视频路径
    video_path = root_dir + '/' + video_path

    cut_range = item.get('cut')  # 获取cut范围

    if video_path and cut_range and len(cut_range) == 2:
        start_frame, end_frame = cut_range
        if os.path.exists(video_path):  # 确保视频文件存在
            try:
                # 获取视频文件名并添加起始帧和结束帧到文件名
                video_name = os.path.basename(video_path)
                base_name, ext = os.path.splitext(video_name)
                new_video_name = f"{base_name}_{start_frame}_{end_frame}{ext}"

                # 拼接新的文件路径
                destination_path = os.path.join(destination_folder, new_video_name)

                # 使用ffmpeg截取视频片段
                (
                    ffmpeg
                    .input(video_path)
                    .filter_('select', f'between(n,{start_frame},{end_frame})')
                    .output(destination_path, codec='libx264', format='mp4')
                    .run(overwrite_output=True)
                )

                # 更新path为新路径
                item['path'] = os.path.abspath(destination_path)
                print(f"成功切割并保存: {video_path} -> {destination_path}")
                return item
            except Exception as e:
                print(f"处理失败: {video_path}, 错误: {e}")
                return None
        else:
            print(f"无效的文件路径: {video_path}")
            return None
    else:
        print(f"缺少cut范围或格式不正确: {item}")
        return None

# 使用多进程处理视频
updated_data = []
with ProcessPoolExecutor(max_workers=32) as executor:
    futures = {executor.submit(process_video, item, root_dir, destination_folder): item for item in selected_data}
    for future in as_completed(futures):
        result = future.result()
        if result:
            updated_data.append(result)

# 将更新后的数据保存为新的json文件
with open('/storage/hxy/ID/Consis-multiID/jsons/updated_data.json', 'w') as f:
    json.dump(updated_data, f, ensure_ascii=False, indent=4)