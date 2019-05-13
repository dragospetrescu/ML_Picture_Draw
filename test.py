import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage.draw import line


########## READ IMAGE
# image = plt.imread("image1.png", format="RGBA")
# image.setflags(write=1)
# for i in range(0, len(image)):
#     line = image[i]
#     for j in range(0, len(line)):
#         pixel = line[j]
##################################

########## SAVE IMAGE
# img = Image.fromarray(image)
# img.save("rgba.png")
####################################

############ ADD SHAPE TO IMAGE
#rr, cc = line(1, 1, 8, 8)
#img[rr, cc] = 1
###################################

def fitness(image1, image2):
    score = 0.0
    for i in range(0, len(image1)):
        line1 = image1[i]
        line2 = image2[i]
        for j in range(0, len(line1)):
            pixel1 = line1[j]
            pixel2 = line2[j]
            score += np.sum(abs(pixel1 - pixel2))
    return score

image1 = plt.imread("image1.png", format="RGB")
# image2 = plt.imread("image1.png", format="RGB")
# score = fitness(image1, image2)
print(image1)