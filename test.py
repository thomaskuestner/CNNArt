
# SOLUTION A: patches match
# STOP in debugger line 796 in data.py
# voxel_ndarray #from line 796 in data.py
f = h5py.File('/home/s1304/no_backup/s1304/output/tmp/Patients-2_Datasets-6_3D_SegMask_64x64x16_Overlap-0.8_Labeling-maskSplit-patientCrossVal/datasets_01_ab02_az_oldhdf5.h5')
X_test_h5 = np.array(f.get('X_test'))
X_test_curr = X_test_h5[test_set_patch:test_set_patch+test_set_currpatch, :, :, :, :]  # index into array given by line 824 in data.py

X_test_dPatches, dLabels = fRigidPatching3D_maskLabeling(voxel_ndarray,
                                                  [self.patchSizeX, self.patchSizeY,
                                                   self.patchSizeZ],
                                                  self.patchOverlap,
                                                  labelMask_ndarray,
                                                  0.5,
                                                  dataset)
# compare visually or it must hold that: X_test_dPatches == X_test_curr!!!!


# SOLUTION B: images match
# or check if the unpatched images match:
# however this check already assumes to get the actual image size from the corresponding DICOM image
# nevertheless you can see if this creates an interpretable unpatched image

##X_test_unpatched = fUnpatchSegmentation(X_test_curr,
                     patchSize=self.patchSizePrediction,
                     patchOverlap=self.patchOverlapPrediction,
                     actualSize=dicom_size,
                     iClass=1)

# compare visually or it must hold that: X_test_unpatched == voxel_ndarray!!!!!