task=${1}

python scripts/train_drqv2.py task=${task} reward=viper reward.expl_std=true reward.expl_update_interval=1