"""Optical Flow for the images in the set.

Optical flow is used to find the pattern of the motion of the objects between two consecutive frames. This implementation
follows the Dense Optical Flow method, which tracks the whole objects, instead of jus borders. For finding the gradients,
opencv uses Gunner Farneback's algorithm. The implementation uses the opencv api for the algorithm implementation. 

The output is saved in the <image_source_path>_opt folder with the same image name.


Example:
    How to run::

        $ python optical_flow.py -a<Source_Path>




References: 

Gunner Farneback's algorithm : http://www.diva-portal.org/smash/get/diva2:273847/FULLTEXT01.pdf

Open CV documentation :
https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html,
https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html,
https://github.com/opencv/opencv/blob/master/samples/python/opt_flow.py

os references: https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist,
https://stackoverflow.com/questions/19932130/python-iterate-through-folders-then-subfolders-and-print-filenames-with-path-t/19932441

"""
import cv2
import numpy as np
import os
import argparse


def draw_hsv(flow):
    """
    Given the calculated flow variable from the opencv method, this method create an image
    showing just the area which are being changes or were in motion, and rest is just blank. 
    HSV: Heu, Saturation, Value is the standard method in image processing for depicting an image. 
    From the calcuated gradient values, the method tries to create an image.
    """
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


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
This block iterates through all the images in sequence to find the optical flow and then stores the flow images 
in the end.

The newly found filtered images are stored in the new folders named after appending _opt folder.
"""
for root,dirs,files in os.walk(path):
    for directory in dirs:
        print('Dir:' + directory)
        for dir_root,dir_inner,dir_files in os.walk(os.path.join(path,directory)):
            #Sort all the files in sequence.
            #This works well with current dataset.
            dir_files = sorted(dir_files)
            
            #Get the current images and convert it into Gray scale and then get the hsv co-ordinates.
            #Even though the image is already in gray scale, removing the cv2.cvtColor would give error.
            old_img = cv2.imread(os.path.join(dir_root,dir_files[0]))
            old = cv2.cvtColor(old_img,cv2.COLOR_BGR2GRAY)
            hsv = np.zeros_like(old_img)
            hsv[...,1] = 255
            
            #output path
            #create the folder,if it is already not there.
            op_path = dir_root + "_opt"
            try:
                if not os.path.exists(op_path):
                    os.makedirs(op_path)
            except OSError as e:
                raise
            
            for file in dir_files:
                #Start with the 0001.png file.
                if file.endswith('.png') and ( not file.startswith('m') and (not '0000' in file)):
                    print('File in process:' + os.path.join(dir_root,file))
                    #Get the new image and convert to gray scale.
                    new_img = cv2.imread(os.path.join(dir_root,file))
                    new = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
                    
                    #Find the gradient values, with default parameters given in the open cv implementation.
                    flow = cv2.calcOpticalFlowFarneback(old,new, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    #New image is now old.
                    old_img = new_img
                    #Write the files.
                    cv2.imwrite(os.path.join(op_path,file),draw_hsv(flow))
                        
#Finishing script.
cv2.destroyAllWindows()
        
            
