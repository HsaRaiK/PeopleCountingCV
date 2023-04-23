import cv2 as cv
#  pip install opencv-python
# to install OpenCV
import os
import basicgui

def main():
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

        # reading from frame
        ret, frame = camera.read()

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