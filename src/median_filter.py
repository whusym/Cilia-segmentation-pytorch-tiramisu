"""Median Filter for the images in the set.

Median Filter is an image filter which replaces the series of pixel values with the median value under the kernel area.
This program uses the opencv's implementation of median filter with two kernel sizes 3 and 5. The program expects the 
input path as an argument, where all the images are either in directory or subdirectory exists. 

The output is saved in the <image_source_pat>_m3 or <image_source_pat>_m5 folder respective to the kernel size.  


Example:
    How to run::

        $ python median_filter.py -a<Source_Path>




References: 

Median Filter Wiki : https://en.wikipedia.org/wiki/Median_filter
Open CV documentation : https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html,
https://docs.opencv.org/3.1.0/d4/d86/group__imgproc__filter.html#ga564869aa33e58769b4469101aac458f9
os references: https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist,
https://stackoverflow.com/questions/19932130/python-iterate-through-folders-then-subfolders-and-print-filenames-with-path-t/19932441

"""
import cv2
import numpy as np
import os


"""
Set the argument for the source path.
"""
parser = argparse.ArgumentParser(description='P4- Celia Segmentation')
parser.add_argument('-a','--source_path', type=str,
                    help='Source Path')


"""
Parse the argument for the source path.
"""
args = vars(parser.parse_args())
print(args)
path = args['source_path']



"""
This block iterates through all the images with png format in the directory or subdirectory under the source path. 
Also, it skips iterating through the mask file. The open cv's medianBlur method applies the filter to the image, with
two kernels sized 3 and 5. 

The newly found filtered images are stored in the new folders named after appending _m3 or _m5 to the original folder.
"""
for root,dirs,files in os.walk(path):
    for file in files:
        if file.endswith('.png') and ( not file.startswith('m')):
            
            #Apply median filter of kernel size 5.
            img = cv2.imread(os.path.join(root,file))
            medianImg = cv2.medianBlur(img,5)
            
            
            #Save the file. Create the required directory if does not exist.
            op_path5 = root + "_m5"
            try:
                if not os.path.exists(op_path5):
                    os.makedirs(op_path5)
            except OSError as e:
                raise
            
            cv2.imwrite(os.path.join(op_path5,file),medianImg)
            
            
            #Apply median filter of kernel size 3.
            medianImg = cv2.medianBlur(img,3)
            op_path3 = root + "_m3"
            
            #Save the file. Create the required directory if does not exist.
            try:
                if not os.path.exists(op_path3):
                    os.makedirs(op_path3)
            except OSError as e:
                raise    
            cv2.imwrite(os.path.join(op_path3,file),medianImg)
        
