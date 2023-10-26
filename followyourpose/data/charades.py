import os
import random
import json
from abc import abstractmethod
import math
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from decord import VideoReader, cpu
import torchvision.transforms._transforms_video as transforms_video

class CharadesDataset(Dataset):
    def __init__(self, video_path, width=512, height=512, n_sample_frames=8, dataset_set='train', prompt=None, sample_frame_rate=2, sample_start_idx=0, accelerator=None):
        try:
            host_gpu_num = accelerator.num_processes
            host_num = 1
            all_rank = host_gpu_num * host_num
            global_rank = accelerator.local_process_index
        except:
            pass
        print('dataset rank:', global_rank, ' / ',all_rank, ' ')
        
        # initialise object
        # self.video_dir = 'pose_example/video_clips'
        self.video_dir = video_path
        self.meta_path = 'followyourpose/data/meta_data.json'
        spatial_transform = 'resize_center_crop'
        resolution=width
        load_raw_resolution=True

        video_length= n_sample_frames
        fps_max=None
        load_resize_keep_ratio=False
        self.global_rank = global_rank
        self.all_rank = all_rank
        # self.subsample = subsample
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.frame_stride = sample_frame_rate
        self.load_raw_resolution = load_raw_resolution
        self.fps_max = fps_max
        self.load_resize_keep_ratio = load_resize_keep_ratio
        self._load_metadata()
        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms_video.RandomCropVideo(crop_resolution)
            elif spatial_transform == "resize_center_crop":
                assert(self.resolution[0] == self.resolution[1])
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(resolution),
                    transforms_video.CenterCropVideo(resolution),
                    ])
            elif spatial_transform == "center_crop":
                self.spatial_transform = transforms_video.CenterCropVideo(resolution)
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None

    def _load_metadata(self):
        # Initialize an empty list to store metadata
        self.metadata = []

        # Define the path to the JSON file
        json_path = self.meta_path

        try:
            # Open the JSON file and load its contents
            with open(json_path, 'r', encoding='utf-8') as json_file:
                metadata_list = json.load(json_file)
                # track number of videos
                self.n_videos = len(metadata_list)

                # Iterate through each entry in the JSON file
                for entry in metadata_list:
                    video_id = entry.get("id", "")
                    # caption = entry.get("script", "")
                    actions = entry.get("actions", [])
                    
                    if video_id and actions:
                        video_part = 1
                        for action in actions:
                            action_label = action.get("action", "")
                            video_part_id = video_id + str(video_part)
                            self.metadata.append({
                                "video_id": video_id,
                                "video_part_id": video_part_id,
                                "action": action_label,
                            })
                            video_part += 1

        except Exception as e:
            # Handle exceptions (e.g., file not found or JSON parsing error)
            print(f"Error loading metadata from JSON: {str(e)}")

        # Optionally, you can print the loaded metadata to verify
        print(f"Loaded {len(self.metadata)} metadata entries.")

    def __len__(self):
        return len(self.metadata)
    
    def _get_video_path(self, video):
        video_sub_path = video['video_part_id'] + '.mp4'
        video_path = os.path.join(self.video_dir, video['video_id'], video_sub_path)
        return video_path
    
    def __getitem__(self, index):
        while True:
            
            index = index % len(self.metadata)
            video = self.metadata[index]
            video_path = self._get_video_path(video)
             
            try:
                if self.load_raw_resolution:
                    video_reader = VideoReader(video_path, ctx=cpu(0))
                elif self.load_resize_keep_ratio:
                    # resize scale is according to the short side
                    h, w, c = VideoReader(video_path, ctx=cpu(0))[0].shape
                    if h < w:
                        scale = h / self.resolution[0]
                    else:
                        scale = w / self.resolution[1]

                    h = math.ceil(h / scale)
                    w = math.ceil(w / scale)
                    video_reader = VideoReader(video_path, ctx=cpu(0), width=w, height=h)
                else:
                    video_reader = VideoReader(video_path, ctx=cpu(0), width=self.resolution[1], height=self.resolution[0])
                if len(video_reader) < self.video_length:
                    print(f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
                continue
            fps_ori = video_reader.get_avg_fps()

            fs = self.frame_stride
            allf = len(video_reader)
            if self.frame_stride != 1:
                all_frames = list(range(0, len(video_reader), self.frame_stride))
                if len(all_frames) < self.video_length:
                    fs = len(video_reader) // self.video_length
                    assert(fs != 0)
                    all_frames = list(range(0, len(video_reader), fs))
            else:
                all_frames = list(range(len(video_reader)))
            
            # select a random clip
            rand_idx = random.randint(0, len(all_frames) - self.video_length)
            frame_indices = all_frames[rand_idx:rand_idx+self.video_length]
            try:
                frames = video_reader.get_batch(frame_indices)
                break
            except:
                print(f"Get frames failed! path = {video_path}")
                index += 1
                continue

        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
       
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
        assert(frames.shape[2] == self.resolution[0] and frames.shape[3] == self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        frames = frames.byte()
        # fps
        fps_clip = fps_ori // self.frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max
        
        # # caption index
        # middle_idx = (rand_idx + self.video_length /2 )*fs
        # big_cap_idx = (middle_idx // 64 +1) *64
        # small_cap_idx = (middle_idx // 64) *64
        # if big_cap_idx >= allf or ((big_cap_idx-middle_idx) >= (small_cap_idx-middle_idx)):
        #     cap_idx = small_cap_idx
        # else:
        #     cap_idx = big_cap_idx
        # # print(middle_idx, small_cap_idx, big_cap_idx,cap_idx)
        # caption = video[1][int(cap_idx//64)]
        print('caption:', video['action'])

        frames = frames.permute(1,0,2,3)
        skeleton_final = torch.zeros_like(frames).byte()
        frames = (frames / 127.5 - 1.0)
        skeleton_final = (skeleton_final / 127.5 - 1.0)
        output = {'pixel_values': frames, 'sentence': video['action'], 'pose': skeleton_final}

        return output