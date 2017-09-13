#utility for resizing images in a directory. The result images will be square of dim=(size x size)
#syntax: python resize_imgs.py directory size(default 64)
import os
import sys
from PIL import Image

def resize(directory, fileName, size):
    filePath = os.path.join(directory, fileName)
    im = Image.open(filePath)
    dirName = directory + "/resized" + str(size) + "/"
    if not os.path.exists(dirName):
        os.makedirs(dirName) 
    im.save(dirName + fileName[:-3]+"png")   
    w, h  = im.size

    newIm = im.resize((int(size),int(size)), resample=Image.LANCZOS)

    newIm.save(dirName + fileName[:-3]+"png")


def bulkResize(imageDirectory, size):
    imgExts = ["png", "bmp", "jpg"]
    for file in os.listdir(imageDirectory):

        ext = file[-3:].lower()

        if ext not in imgExts:
            continue

        resize(imageDirectory, file, size)

if __name__ == "__main__":
    if(len(sys.argv) < 2):
        print ('Please specify a directory')
    else :
        imageDirectory=sys.argv[1] # first arg is path to image directory
        if(len(sys.argv) > 2):
            size=sys.argv[2]
        else:
            size=64

        bulkResize(imageDirectory, size)
        print ("done")
