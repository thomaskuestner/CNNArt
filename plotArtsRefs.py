import matplotlib.pyplot as plt
import numpy as np

index = 100
artimg = "D:/med_data/MRPhysics/DeepLearningArt_GAN/ArtGAN_1Patients_P100x100_O0.5/data_arts/Art" + str(index) + ".npy"
refimg = "D:/med_data/MRPhysics/DeepLearningArt_GAN/ArtGAN_1Patients_P100x100_O0.5/data_refs/Ref" + str(index) + ".npy"

art = np.load(artimg)

ref = np.load(refimg)

plt.subplot(121)
plt.imshow(art, cmap='gray')
plt.subplot(122)
plt.imshow(ref, cmap='gray')

print()