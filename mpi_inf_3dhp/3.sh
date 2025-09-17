#!/bin/bash
#SBATCH --job-name=PoseformerV2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=12
#SBATCH --partition=P2
#SBATCH --output=./logs/%A_out.log

# python run_energy_poseformer.py -g 0123 -k cpn_ft_h36m_dbb \
#   -frame 27 -frame-kept 3 -coeff-kept 3 -c /home/s3/kimyeonsung/PoseFormerV2/checkpoint_3dhp\
#   --energy-weight 1e-5 --lr-loss 1e-4 --em-loss-type margin --em-margin-type mpjpe \
#   > out_3dhp.log 2> error_3dhp.log


# python -u run_3dhp.py --gpu 0 \
#     -f 27 -frame-kept 3 -coeff-kept 3 \
#     --train 1 --lr 0.0007 -lrd 0.97 \
#     -c checkpoint_3dhp \
#     > out_3dhp.log 2> error_3dhp.log

#원래 학습하던거

python -u run_3dhp.py --gpu 0 \
    -f 27 -frame-kept 3 -coeff-kept 3 \
    --train 1 --lr 0.0007 -lrd 0.97 \
    -c checkpoint_3dhp \
    --resume latest_epoch.pth \
    >> out_3dhp.log 2>> error_3dhp.log
#이어서 학습하는것


python -u run_3dhp.py --gpu 0 \
    -f 27 -frame-kept 3 -coeff-kept 3 \
    --train 1 --lr 0.0007 -lrd 0.97 \
    -c checkpoint_3dhp \
    > out_3dhp.log 2> error_3dhp.log