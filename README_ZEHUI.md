## config
- in main.py to choose which config should be chosen (param_2D.yml/param_3D.yml) according to network dimension.


## new
- ../utils/MotionCorrection/network_block.py add 3D structure
- ../utils/MotionCorrection/network.py define 3D network
- ../utils/MotionCorrection/customLoss.py define new loss functions
- ../correction/networks/motion/VAE2D/motion_VAE2D.py add 2D MS_SSIM_loss
- ../correction/networks/motion/VAE3D/motion_VAE3D.py add 3D MS_SSIM_loss

