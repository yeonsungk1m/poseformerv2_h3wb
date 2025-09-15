#!/bin/bash
#SBATCH --job-name=PoseformerV2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --partition=P2
#SBATCH --output=./logs/%A_out.log
#SBATCH --error=./logs/%A_error.log

# python run_energy_poseformer.py -g 0 -k cpn_ft_h36m_dbb \
#   -frame 27 -frame-kept 3 -coeff-kept 3 -c /home/s3/kimyeonsung/PoseFormerV2/checkpoint \
#   --energy-weight 1e-5 --lr-loss 1e-4 --em-loss-type margin --em-margin-type mpjpe \
#   > out.log 2> error.log

python -u run_3dhp.py --gpu 0 \
    -f 27 -frame-kept 3 -coeff-kept 3 \
    --train 1 --lr 0.0007 -lrd 0.97 \
    -c checkpoint_3dhp_seal\
    > out_3dhp_seal.log 2> error_3dhp_seal.log