
from app_followyourpose import *

import copy
import gradio as gr
from transformers import AutoTokenizer, CLIPTextModel
from huggingface_hub import snapshot_download
from inference_mmpose import *
import sys
sys.path.append('FollowYourPose')

def get_time_string() -> str:
    x = datetime.datetime.now()
    return f"{(x.year - 2000):02d}{x.month:02d}{x.day:02d}-{x.hour:02d}{x.minute:02d}{x.second:02d}"


class merge_config_then_run():
    def __init__(self) -> None:
            # Load the tokenizer
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.unet = None
        # self.download_model()
        self.mmpose = gr.Interface.load(name="spaces/fffiloni/mmpose-estimation")
    
    # def download_model(self):
    #     REPO_ID = 'YueMafighting/FollowYourPose_v1'
    #     snapshot_download(repo_id=REPO_ID, local_dir='./checkpoints/', local_dir_use_symlinks=False)     
 
            
    def run(
        self,
        data_path,
        target_prompt,
        num_steps,
        guidance_scale,
        video_type,
        user_input_video=None,
        start_sample_frame=0,
        n_sample_frame=8,
        stride=1,
        left_crop=0,
        right_crop=0,
        top_crop=0,
        bottom_crop=0,
    ):
        if video_type == "Raw Video":
            infer_skeleton(self.mmpose, data_path)
        default_edit_config='./configs/pose_sample.yaml'
        Omegadict_default_edit_config = OmegaConf.load(default_edit_config)
        
        dataset_time_string = get_time_string()
        config_now = copy.deepcopy(Omegadict_default_edit_config)

        offset_dict = {
            "left": left_crop,
            "right": right_crop,
            "top": top_crop,
            "bottom": bottom_crop,
        }
        ImageSequenceDataset_dict = {
            "start_sample_frame" : start_sample_frame,
            "n_sample_frame" : n_sample_frame,
            "sampling_rate"       : stride,   
            "offset": offset_dict,
        }
        config_now['validation_data'].update(ImageSequenceDataset_dict)
        if user_input_video and data_path is None:
            raise gr.Error('You need to upload a video or choose a provided video')
        if user_input_video is not None:
            if isinstance(user_input_video, str):
                config_now['validation_data']['path'] = user_input_video
            elif hasattr(user_input_video, 'name') and user_input_video.name is not None:
                config_now['validation_data']['path'] = user_input_video.name
        config_now['validation_data']['prompts'] = [target_prompt]
        # ddim config
        config_now['validation_data']['guidance_scale'] = guidance_scale
        config_now['validation_data']['num_inference_steps'] = num_steps
        
        if video_type == "Raw Video":
            config_now['skeleton_path'] = './mmpose_result.mp4'
        else:
            config_now['skeleton_path'] = data_path
        
        save_path = test(**config_now)
        mp4_path = save_path.replace('_0.gif', '_0_0_0.mp4')
        return mp4_path

