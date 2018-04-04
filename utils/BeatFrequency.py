import numpy as np
import numpy.fft as fft
from skimage import data_dir,io,color
import matplotlib
import sys

#read 100 png from folder
coll = io.ImageCollection('frame*.png')
#generate 3 dim array [Frame, W, H]
mat=io.concatenate_images(coll)
#compute the actual  at each pixel
mean = mat- mat.mean(axis = 0)
mat_fft = fft.fft(mean, n = 100, axis = 0) / 100
mat_abs = 2 * np.absolute(mat_fft[:51])
#find beat frequency for each pixel
max_freq_indices = mat_abs.argmax(axis = 0)
#generate dominant array for each pixel
merge = np.zeros((max_freq_indices.shape[0],max_freq_indices.shape[1]), dtype=np.int16)
for i in range(max_freq_indices.shape[0]):
	for j in range(max_freq_indices.shape[1]):
		merge[i][j] = mat[max_freq_indices[i][j]][i][j]
#save png
matplotlib.image.imsave(sys.argv[1]+'.png', merge)
