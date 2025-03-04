import decord
import json
import cv2
import os

# def draw_bbox_and_track_id(video_path, json_data, output_path, cut):
#     # 打开视频文件
#     cap = cv2.VideoCapture(video_path)
    
#     # 获取视频的基本信息
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
#     # 创建视频写入器
#     out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
#     # 遍历每一帧
#     frame_count = cut[0]
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # 获取当前帧的数据（假设 JSON 中的键是帧号）
#         frame_data = json_data.get(frame_count)

#         import ipdb;ipdb.set_trace()

#         if frame_data:
#             for face in frame_data["face"]:
#                 # 获取 bbox 和 track_id
#                 track_id = face["track_id"]
#                 box = face["box"]
#                 x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])

#                 # 绘制 bbox（矩形框）
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框

#                 # 在 bbox 上方添加 track_id 文本
#                 cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#             # 写入处理后的帧到输出视频
#         out.write(frame)

#         # 增加帧计数
#         frame_count += 1

#     # 释放资源
#     cap.release()
#     out.release()
#     print(f"处理完成，结果已保存到: {output_path}")


def draw_bbox_and_track_id(video_path, json_data, output_path, cut):
    """
    读取视频并在指定帧范围内绘制 bbox 和 track_id，只写入该范围内的帧到新视频。
    
    :param video_path: 输入视频路径
    :param json_data: 包含帧数据的 JSON
    :param output_path: 输出视频路径
    :param cut: [start_frame, end_frame]，指定要写入的帧范围
    """
    cap = cv2.VideoCapture(video_path)

    # 获取视频信息
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)  # 视频帧率
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建视频写入器
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # 获取起始和结束帧
    start_frame, end_frame = cut

    # 计数当前帧号
    frame_count = 0

    # import ipdb;ipdb.set_trace()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 只处理指定范围的帧
        if start_frame <= frame_count <= end_frame:
            frame_data = json_data.get(str(frame_count))

            # import ipdb;ipdb.set_trace()

            if frame_data:
                for face in frame_data["face"]:
                    # 获取 bbox 和 track_id
                    track_id = face["track_id"]
                    box = face["box"]
                    x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])

                    # 绘制 bbox（矩形框）
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框

                    # 在 bbox 上方添加 track_id 文本
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # 写入处理后的帧到输出视频
            out.write(frame)

        # 如果超过 `end_frame`，跳出循环
        if frame_count > end_frame:
            break

        frame_count += 1  # 递增帧计数

    # 释放资源
    cap.release()
    out.release()
    print(f"处理完成，仅处理 {start_frame} ~ {end_frame} 帧，结果已保存到: {output_path}")



if __name__ == "__main__":

    input_jsons_dir = '/storage/hxy/ID/data/data_processor/verification_jsons/2'
    input_videos_dir = '/storage/hxy/ID/data/data_processor/verification'
    video_root = "/storage/hxy/ID/data/data_processor/test"
    output_dir = '/storage/hxy/ID/data/data_processor/verification/2'
    

    jsons_files = [f for f in os.listdir(input_jsons_dir) if f.endswith(".json")]
    video_files = [f for f in os.listdir(input_videos_dir) if f.endswith(".mp4")]

    for json_file in jsons_files:

        # json_file = '/storage/hxy/ID/data/data_processor/test/step2_jsons/sucai/gm1464771122-497372401_part1_step1-162-261.json'


        with open(os.path.join(input_jsons_dir, json_file), 'r') as f:
            json_data = json.load(f)



        metadata = json_data['metadata']
        bbox = json_data['bbox']

        # video_path = video_root + '/' + metadata['path']
        # video_path = '/storage/hxy/ID/data/data_processor/test/step2_test_videos/gm1464771122-497372401_part1.mp4'
        output_path = os.path.join(output_dir, json_file.replace('.json', '.mp4'))

        #test input
        # json_path = '/storage/hxy/ID/data/data_processor/test/wrong_examples/frames_gm621059908/gm621059908-108728121_part2_0-97.json'
        # with open(json_path, 'r') as f:
        #     json_data = json.load(f)

        # video_path = '/storage/hxy/ID/data/data_processor/test/wrong_examples/gm621059908-108728121_part2.mp4'
        # bbox = json_data['bbox']
        # metadata = json_data['metadata']
        # output_path = '/storage/hxy/ID/data/data_processor/test/wrong_examples/frames_gm621059908/4/gm621059908-108728121_part2_0-97.mp4'

        # metadata['cut'] = [0,97]

        # import ipdb;ipdb.set_trace()

        #test video dir
        video_path = os.path.join(input_videos_dir, json_file.replace('.json', '.mp4'))
        metadata = {}
        metadata['cut'] = [0, len(bbox)]

        draw_bbox_and_track_id(video_path, bbox, output_path, metadata['cut'])

        # break 


