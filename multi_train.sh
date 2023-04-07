

TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch \
    --multi_gpu --num_processes=8 --gpu_ids '0,1,2,3,4,5,6,7' \
    train_followyourpose.py \
    --config="configs/pose_train.yaml" \
    --skeleton_path="./pose_example/vis_ikun_pose2.mov" \


