'''
This is the python file used for postprocessing of the results. Based on our experiment, the best
performance of single model carries an IoU score 40.8. But since we have about two dozens models with
scores over 37, I stack these results together (i.e. add these image arrays together) and set a threshold
to generate an ensembled result out of them. Here is the python scipt for it. Just put these script into
the folder that store all the results (in different folders), and the script will put all processed images
into a new /concatResults folder.
'''

from glob import glob
from numpy import array
from imageio import imread, imwrite
import numpy as np

res_glob = sorted(glob('p40401-1998/*.png'))
res_glob_imgs = array([imread(f) for f in res_glob])
res_glob_2 = sorted(glob('p40401-1751/*.png'))
# print (res_glob_2[0][12:])
res_glob_imgs_2 = array([imread(f) for f in res_glob_2])
res_glob_3 = sorted(glob('p40401-1332/*.png'))
res_glob_imgs_3 = array([imread(f) for f in res_glob_3])
res_glob_4 = sorted(glob('p40401-1748/*.png'))
res_glob_imgs_4 = array([imread(f) for f in res_glob_4])
res_glob_5 = sorted(glob('p40402-1642/*.png'))
res_glob_imgs_5 = array([imread(f) for f in res_glob_5])
res_glob_6 = sorted(glob('p40402-1645/*.png'))
res_glob_imgs_6 = array([imread(f) for f in res_glob_6])
res_glob_7 = sorted(glob('p40404-1232/*.png'))
res_glob_imgs_7 = array([imread(f) for f in res_glob_7])
res_glob_8 = sorted(glob('p40402-1783/*.png'))
res_glob_imgs_8 = array([imread(f) for f in res_glob_8])
res_glob_9 = sorted(glob('p0403-3201/*.png'))
res_glob_imgs_9 = array([imread(f) for f in res_glob_9])
res_glob_10 = sorted(glob('p0404-2473/*.png'))
res_glob_imgs_10 = array([imread(f) for f in res_glob_10])
res_glob_11 = sorted(glob('p0404-4000/*.png'))
res_glob_imgs_11 = array([imread(f) for f in res_glob_11])
res_glob_12 = sorted(glob('p0404-0395/*.png'))
res_glob_imgs_12 = array([imread(f) for f in res_glob_12])
res_glob_13 = sorted(glob('p0404-0393/*.png'))
res_glob_imgs_13 = array([imread(f) for f in res_glob_13])
res_glob_14 = sorted(glob('p0404-0386/*.png'))
res_glob_imgs_14 = array([imread(f) for f in res_glob_14])
res_glob_15 = sorted(glob('p0405-1256/*.png'))
res_glob_imgs_15 = array([imread(f) for f in res_glob_15])
res_glob_16 = sorted(glob('p0405-1507/*.png'))
res_glob_imgs_16 = array([imread(f) for f in res_glob_16])
res_glob_17 = sorted(glob('p0405-2229/*.png'))
res_glob_imgs_17 = array([imread(f) for f in res_glob_17])
res_glob_18 = sorted(glob('p0405-2332/*.png'))
res_glob_imgs_18 = array([imread(f) for f in res_glob_18])
res_glob_19 = sorted(glob('p0405-2370/*.png'))
res_glob_imgs_19 = array([imread(f) for f in res_glob_19])
res_glob_20 = sorted(glob('p0405-2552/*.png'))
res_glob_imgs_20 = array([imread(f) for f in res_glob_20])
res_glob_21 = sorted(glob('p0405-2609/*.png'))
res_glob_imgs_21 = array([imread(f) for f in res_glob_21])
res_glob_22 = sorted(glob('p0405-2775/*.png'))
res_glob_imgs_22 = array([imread(f) for f in res_glob_22])
res_glob_23 = sorted(glob('p0401-2403/*.png'))
res_glob_imgs_23 = array([imread(f) for f in res_glob_23])
concat_imgs = [res_glob_imgs[i] + res_glob_imgs_2[i] + res_glob_imgs_3[i] + res_glob_imgs_4[i] + res_glob_imgs_5[i] + res_glob_imgs_6[i] + res_glob_imgs_7[i] + res_glob_imgs_8[i] + res_glob_imgs_9[i] + res_glob_imgs_10[i] + res_glob_imgs_11[i] + res_glob_imgs_12[i] + res_glob_imgs_13[i] + res_glob_imgs_14[i] \
                     + res_glob_imgs_15[i] + res_glob_imgs_16[i] + res_glob_imgs_17[i] + res_glob_imgs_18[i] + res_glob_imgs_19[i] + res_glob_imgs_20[i] + res_glob_imgs_21[i] + res_glob_imgs_22[i] + res_glob_imgs_23[i] for i in range(len(res_glob_imgs))]

for i in range(len(concat_imgs)):
    concat_imgs[i][concat_imgs[i] <= 15] = 0
    concat_imgs[i][concat_imgs[i] != 0] = 2
    # concat_imgs[i][concat_imgs[i] <= 2] = 0
    imwrite('concatResults/' + res_glob_2[i][12:], concat_imgs[i].astype(np.uint8))
