
TORCH_DISTRIBUTED_DEBUG=DETAIL /apdcephfs/private_mayuema/envs/newtuneavideo/bin/accelerate launch \
    --gpu_ids '0' \
    txt2video.py \
    --config="configs/pose_sample.yaml" \
    --skeleton_path="./pose_example/vis_ikun_pose2.mov" \

