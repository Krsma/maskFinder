import cv2 
import matplotlib.pyplot as plt
import pathlib
import glob
import os

imagePath = "maske.jpg"
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def cutFaces(faces, image, imageName, hunt = False):
    faceNumber = 0
    for (x,y,w,h) in faces:
        roi = image[y:y+h, x:x+w]
        roi = cv2.resize(roi, (128,128))
        #plt.imshow(roi)
        #plt.show()
        #input()
        exportName = os.path.join("export","face-{0}-{1}.png".format(imageName, faceNumber))
        if(hunt == True):
            exportName = os.path.join("export", "hunt", imageName, "face-{0}-{1}.png".format(imageName, faceNumber))
            for (x,y,w,h) in faces:
                imageCopy = image
                cv2.rectangle(imageCopy, (x, y), (x+w, y+h), (0, 255, 0), 2)
                plt.imshow(imageCopy)
                plt.show()
                if(input("Keep? y/n") == 'n'):
                    break
        #exportName = "export/face-{0}-{1}.png".format(imageName, faceNumber)
        faceNumber = faceNumber + 1
        print("Writing image {0}".format(exportName))
        cv2.imwrite(exportName, roi)
def findFaces(imagePath, scaleFactor=1.05):
    image = cv2.imread(imagePath)
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imageGray, scaleFactor, 4)
    imageName = pathlib.PurePath(imagePath).name.split('.')[0]
    print("Found {0} faces for image {1}".format(len(faces), imageName))
    if(len(faces) > 0):
        cutFaces(faces, image, imageName)
    return len(faces)
def huntFace(imagePath, scale = 1.1):
    scale = 1.1
    found = findFaces(imagePath, scale)
    if(input("Try again? y/n") == "y"):
        huntFace(imagePath, scale + 0.1)   
def loadSingle(imagePath):
    findFaces(imagePath)    
if(__name__ == "__main__"):
    totalFaces = 0
    emptyPictures = []
    print("Put photos you want processed into a folder named import")
    print("For folder loading leave input empty")
    PATH = "importSingle/*"
    for x in glob.glob(PATH):
        print("Processing image {0}".format(x))
        facesFound = findFaces(x)
        totalFaces = totalFaces + facesFound
        if(facesFound == 0):
            emptyPictures.append(x)
    print("Analyzed {0} images in total".format(len(glob.glob(PATH))))    
    print("Generated {0} face images in total".format(totalFaces))
    print("{0} images with no faces found \n Empty pictures \n".format(len(emptyPictures)))
    for a in emptyPictures:
        print(a)
    print("Additional attempts on empty pictures?")
    if(input("Try again? y/n") == "y"):
        for x in emptyPictures:
            huntFace(x)
