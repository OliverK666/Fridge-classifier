import os
import glob
from PIL import Image
import numpy as np
import cv2
import random
import fnmatch

# assumung that the names of the images and the labels are same excepth the extension

Dict = {0 : 'apple', 1 : 'banana', 2 : 'beef', 3 : 'blueberries', 4 : 'bread', 5 : 'butter', 6 : 'carrot', 7 : 'cheese', 8 : 'chicken', 9 :'chicken_breast',
        10 : 'chocolate', 11 : 'corn', 12 : 'eggs', 13 : 'flour', 14 : 'goat_cheese', 15 : 'green_beans', 16 : 'ground_beef', 17 : 'ham', 18 : 'heavy_cream', 19 : 'lime',
        20 : 'milk', 21 : 'mushrooms', 22 : 'onion', 23 : 'potato', 24 : 'shrimp', 25 : 'spinach', 26 : 'strawberries', 27 : 'sugar', 28 : 'sweet_potato', 29 : 'tomato'}

images = glob.glob("C://Users//Oli//Documents//Faks//2//seminarski b//aicook.v1-original-images.yolov8//train//images//*")
labels = glob.glob("C://Users//Oli//Documents//Faks//2//seminarski b//aicook.v1-original-images.yolov8//train//labels//*")
dest = ("C://Users//Oli//Documents//Faks//2//seminarski b//aicook.v1-original-images.yolov8//train//bestAugmented256")

images = sorted(images)
labels = sorted(labels)


for i in range(len(images)):
    with open(labels[i], "r") as r:
        label = r.readlines()

    for line in label:
        dest = ("C://Users//Oli//Documents//Faks//2//seminarski b//aicook.v1-original-images.yolov8//train//bestAugmented256")
        value, x, y, w, h = line.split(" ")
        image = cv2.imread(images[i])

        #print(image)

        value = int(value)
        x = float(x) * 1424
        y = float(y) * 2144
        w = float(w) * 1424
        h = float(h) * 2144

        offset = 25


        r1 = random.randint(-offset, offset)
        r2 = random.randint(-offset, offset)
        r3 = random.randint(-offset, offset)
        r4 = random.randint(-offset, offset)

        tlx = max(int(x - w / 2), 0)
        tly = max(int(y - h / 2), 0)
        brx = min(int(x + w / 2), 1424)
        bry = min(int(y + h / 2), 2144)

        #image = np.asarray(image)

        """

        rand = random.randint(25, 50)
        if i % 4 == 0:
            tlx = tlx + int((brx - tlx) * rand / 100)
        elif i % 4 == 1:
            brx = brx - int((brx - tlx) * rand / 100)
        elif i % 4 == 2:
            tly = tly + int((bry - tly) * rand / 100)
        else:
            bry = bry - int((bry - tly) * rand / 100)
            
        """

        image = image[tly:bry, tlx:brx]

        if tlx == brx or tly == bry:
            continue
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)

        #image = cv2.GaussianBlur(image,(3,3),cv2.BORDER_DEFAULT)

        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #print(image.shape)

        #image = Image.fromarray(image)

        folder = Dict[value]

        dest = f"{dest}//{folder}"
        if not os.path.exists(dest):
            os.mkdir(dest)

        count = len(fnmatch.filter(os.listdir(dest), '*.*'))

        dest = f"{dest}//{folder}{i}.jpg"

        print(dest)



        #if count < 30:
        cv2.imwrite(dest, image)
