# CNNArt Motion Correction
This CNN based motion correction approach is implemented with Variational Autoencoder (VAE) and tested on Head, Pelvis, Abdomen and combined datasets.

## Motion Correction training
Make following modifications in config/param.yml (refer to config/param_MC.yml or param_2D.yml/param_3D.yml) and run `python main.py`.

- `selectedDatabase`: select the specific dataset for training
- `range`: [-1, 1]
- `patchSize`: select the patch size for training, currently only [48, 48] or [80, 80] are supported
- `patchOverlap`: select the patch overlapping rate, `0.8` is used in this work
- `sLabeling`: volume
- `sPatching` : rigidPatching
- `sSplitting` : crossvalidation_patient (one patient is selected for testing)
- `batchSize` : [128] for patch size [48, 48]; [64] for patch size [80, 80] \(limited to GPU memory\)
- `lr` : [0.0001] for head and pelvis dataset; [0.00001] for abdomen dataset
- `lTrain` : true
- `lCorrection`: true
- `sCorrection`: motion_VAE2D
- `pl_network`: vgg19
- `arch`: vae-mocounet or vae-moconet
- `kl_weight`: 1  (KL loss weight)
- `tv_weight`: 0 (TV loss weight, no TV loss is calculated)
- `ge_weight`: 10 (gradient entropy weight)
- `perceptual_weight`: 0.00001 (perceptual loss weight)
- `mse_weight`: 0 (MSE loss weight, no MSE loss is calculated)
- `charbonnier_weight`: 0.1 (Charbonnier loss weight)
- `loss_ref2ref`: 0.4 (ref2ref weight)
- `loss_art2ref`: 0.6 (art2ref weight)
- `nScale`: 255
- `test_patient`: select a specific patient for testing and the rest for training
- `augmentation`: false (augmentation is not used in this work due to the time limitation)

## Motion Correction predicting
Make following odifications in config/param.yml based on previous parameter settings and run `python main.py`.

- `lTrain` : false
- `bestModel`: select the trained model weights for predicting
- `actualSize`: the actual unpatched size of the original images
- `evaluate`: true
- `unpatch`: true

## Version control
### 3D VAE
- ../utils/MotionCorrection/network_block.py add 3D structure
- ../utils/MotionCorrection/network.py define 3D network
- ../utils/MotionCorrection/customLoss.py define new loss functions
- ../correction/networks/motion/VAE2D/motion_VAE2D.py add 2D MS_SSIM_loss
- ../correction/networks/motion/VAE3D/motion_VAE3D.py add 3D MS_SSIM_loss
