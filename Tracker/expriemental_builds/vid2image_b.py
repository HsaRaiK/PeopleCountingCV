import cv2 as cv
#  pip install opencv-python
# to install OpenCV
import os
import basicgui
import os
import glob
import shutil
import time

import numpy

workingpath = '/home/hsaraik/workspace/gitv1/Tracker/imagesdata'
# change this to the directory that you want to save pictures in 

def run_vid_2_img():
    camera = cv.VideoCapture(basicgui.filepath)

    try:

        # creating a folder named data
        if not os.path.exists('imagesdata'):
            os.makedirs('imagesdata')

    # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

    # frame
    currentframe = 0

    while (True):
        #if cv.waitKey():
            # reading from frame
            ret, frame = camera.read()


            # Artificially decreasing the number of frames per second using a loop

            if ret:
                # if video is still left continue creating images
                name = './imagesdata/frame' + str(currentframe) + '.jpg'
                print('Creating...' + name)
                # writing the extracted images
                cv.imwrite(name, frame)
                # increasing counter so that it will
                # show how many frames are created
                currentframe += 1
            else:
                break

    # Release all space and windows once done
    camera.release()
    cv.destroyAllWindows()
    # More Jank less bank
    kek, targ = 0, 1
    folder = workingpath

    for filename in os.listdir(folder):
        if(kek == targ):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
            kek = 0
        else:
            kek += 1





    # Janky part of the code only for debugging
    leaveC = int(input("Please choose 1 to exit"))
    #print(leaveC)
    if(leaveC == 1 ):
        folder = workingpath
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

