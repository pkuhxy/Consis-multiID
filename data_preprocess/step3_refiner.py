import json
import os




class Refiner:
    def __init__(self, metadata, bbox_info):
        self.useful_frames = metadata['face_cut']
        self.bbox_info = bbox_info
        self.pool_size = 2
        self.length = len(bbox_info)

        self.uncertain_frames = []

        #获得存在-1的帧的index
        for key, value in self.bbox_info.items():

            for face_info in value['face']:
                if face_info['track_id'] != -1:
                    self.uncertain_frames.append(key)
                    continue



    def count_center(self,bbox):

        x1 = bbox['x1']
        x2 = bbox['x2']
        y1 = bbox['y1']
        y2 = bbox['y2']

        center = [(x1+x2)/2, (y1+y2)/2]

        return center

    def get_frames_center(self, frame):
        frame_centers = {}
        for face_item in self.bbox_info[frame]['face']:
            frame_centers[face_item['track_id']] = self.count_center(face_item['box'])

        return frame_centers





    def refine(self):
        
        for index in self.uncertain_frames:
            forward_index = max(index-1, 0)
            backward_index = min(index+1, self.length-1)

            if forward_index not in self.uncertain_frames and backward_index not in self.uncertain_frames:
                forward_frame_centers = self.get_frames_center(forward_index)
                backward_frame_centers = self.get_frames_center(backward_index)
                
                frame_id = []
                for face_item in self.bbox_info[index]['face']:
                    frame_id.append(face_item['track_id'])

                for face_item in self.bbox_info[index]['face']:

                    if face_item['track_id'] == -1:
                        uncertain_center = self.count_center(face_item['box'])




            if forward_index not in self.uncertain_frames or backward_index not in self.uncertain_frames:
                self.get_real_id(index)

            



