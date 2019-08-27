import numpy as np
def gadget_cnnart(matrix, patch_size, overlap_rate):
    assert len(matrix.shape) <= 3
    if len(matrix.shape) == 3:
        matrix = np.stack((matrix.astype(np.complex64),)*3, axis=0)
        matrix[0, :, :, :] = matrix.max()
        return matrix
    elif len(matrix.shape) == 2:
        matrix = np.stack((matrix.astype(np.complex64),)*3, axis=2)
        half = int(matrix.shape[0]/2)
        matrix[:, half:, 0] = matrix.max()
        matrix[:half, :, 1] = 0
        return matrix