seeds=(406565 925711 558086 53057 506513)
steps=100000

############
# CURL_SAC #
############

for s in ${seeds[@]}; do
    actrepeat=8
    let numsteps=steps/actrepeat
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --domain_name cartpole \
        --task_name swingup \
        --encoder_type pixel \
        --action_repeat $actrepeat \
        --save_tb --save_model --save_video --pre_transform_image_size 100 --image_size 84 \
        --work_dir ./tmp/cartpole \
        --agent curl_sac --frame_stack 3 \
        --seed $s --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq $numsteps --batch_size 128 --num_train_steps $numsteps --init_steps 128
done

for s in ${seeds[@]}; do
    actrepeat=2
    let numsteps=steps/actrepeat
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --domain_name finger \
        --task_name spin \
        --encoder_type pixel \
        --action_repeat $actrepeat \
        --save_tb --save_model --save_video --pre_transform_image_size 100 --image_size 84 \
        --work_dir ./tmp/finger \
        --agent curl_sac --frame_stack 3 \
        --seed $s --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq $numsteps --batch_size 128 --num_train_steps $numsteps --init_steps 128
done

for s in ${seeds[@]}; do
    actrepeat=4
    let numsteps=steps/actrepeat
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --domain_name reacher \
        --task_name easy \
        --encoder_type pixel \
        --action_repeat $actrepeat \
        --save_tb --save_model --save_video --pre_transform_image_size 100 --image_size 84 \
        --work_dir ./tmp/reacher \
        --agent curl_sac --frame_stack 3 \
        --seed $s --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq $numsteps --batch_size 128 --num_train_steps $numsteps --init_steps 128
done

for s in ${seeds[@]}; do
    actrepeat=4
    let numsteps=steps/actrepeat
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --domain_name cheetah \
        --task_name run \
        --encoder_type pixel \
        --action_repeat $actrepeat \
        --save_tb --save_model --save_video --pre_transform_image_size 100 --image_size 84 \
        --work_dir ./tmp/cheetah \
        --agent curl_sac --frame_stack 3 \
        --seed $s --critic_lr 2e-4 --actor_lr 2e-4 --eval_freq $numsteps --batch_size 128 --num_train_steps $numsteps --init_steps 128
done

for s in ${seeds[@]}; do
    actrepeat=2
    let numsteps=steps/actrepeat
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --domain_name walker \
        --task_name walk \
        --encoder_type pixel \
        --action_repeat $actrepeat \
        --save_tb --save_model --save_video --pre_transform_image_size 100 --image_size 84 \
        --work_dir ./tmp/walker \
        --agent curl_sac --frame_stack 3 \
        --seed $s --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq $numsteps --batch_size 128 --num_train_steps $numsteps --init_steps 128
done

for s in ${seeds[@]}; do
    actrepeat=4
    let numsteps=steps/actrepeat
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --domain_name ball_in_cup \
        --task_name catch \
        --encoder_type pixel \
        --action_repeat $actrepeat \
        --save_tb --save_model --save_video --pre_transform_image_size 100 --image_size 84 \
        --work_dir ./tmp/ball_in_cup \
        --agent curl_sac --frame_stack 3 \
        --seed $s --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq $numsteps --batch_size 128 --num_train_steps $numsteps --init_steps 128
done
