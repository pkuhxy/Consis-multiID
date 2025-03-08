import json
import os
import math



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


    def compute_center(self,bbox):

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


    def compute_distance(self, center1, center2):
        delta_x = abs(center1[0] - center2[0])
        delta_y = abs(center1[1] - center2[1])

        delta = math.sqrt(math.pow(delta_x, 2) + math.pow(delta_y, 2))

        return delta


    def refine(self):
        
        while len(self.uncertain_frames)!=0:
            for index in self.uncertain_frames:
                forward_index = max(index-1, 0)
                backward_index = min(index+1, self.length-1)

                if forward_index not in self.uncertain_frames and backward_index not in self.uncertain_frames:
                    forward_frame_centers = self.get_frames_center(forward_index)
                    backward_frame_centers = self.get_frames_center(backward_index)
                    
                    if len(forward_frame_centers) >= len(backward_frame_centers):
                        frame_centers = forward_frame_centers
                    else:
                        frame_centers = backward_frame_centers

                    frame_id = []
                    for face_item in self.bbox_info[index]['face']:
                        frame_id.append(face_item['track_id'])

                    for face_item in self.bbox_info[index]['face']:

                        if face_item['track_id'] == -1:
                            uncertain_center = self.count_center(face_item['box'])
                            min_delta = 9999
                            track_id = -1
                            for face_id, center in frame_centers.items():
                                if face_id in frame_id:
                                    continue
                                
                                delta = self.compute_center(uncertain_center, center)
                                if delta < min_delta:
                                    min_delta = delta
                                    track_id = face_id
                                
                            face_item['track_id'] = track_id
                    self.uncertain_frames.remove(index)
                    continue


                if forward_index not in self.uncertain_frames or backward_index not in self.uncertain_frames:
                    if forward_index not in self.uncertain_frames:
                        frame_centers = self.get_frames_center(forward_index)
                    if backward_frame_centers not in self.uncertain_frames:
                        frame_centers = self.get_frames_center(backward_index)

                    frame_id = []
                    for face_item in self.bbox_info[index]['face']:
                        frame_id.append(face_item['track_id'])

                    for face_item in self.bbox_info[index]['face']:

                        if face_item['track_id'] == -1:
                            uncertain_center = self.count_center(face_item['box'])
                            min_delta = 9999
                            track_id = -1
                            for face_id, center in frame_centers.items():
                                if face_id in frame_id:
                                    continue
                                
                                delta = self.compute_center(uncertain_center, center)
                                if delta < min_delta:
                                    min_delta = delta
                                    track_id = face_id
                                
                            face_item['track_id'] = track_id

                    self.uncertain_frames.remove(index)



