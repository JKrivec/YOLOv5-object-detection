import cv2
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from preprocessing.preprocess import Preprocess


os.chdir(os.path.dirname(os.path.realpath(__file__)))

with open('config.json') as config_file:
	config = json.load(config_file)

images_path = config['images_path']
annotations_path = config['annotations_path']
preprocess = Preprocess()


img_name = images_path + "/0001.png"
img = cv2.imread(img_name)

fig = plt.figure(figsize=(480, 360))


img1 = preprocess.histogram_equlization_rgb(img)
ax1 = fig.add_subplot(3,2, 1)
ax1.title.set_text('histogram_equlization_rgb')
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))

img2 = preprocess.gaussian_threshold(img)
ax2 = fig.add_subplot(3,2, 2)
ax2.title.set_text('gaussian_threshold')
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

img3 = preprocess.sharpen(img)
ax3 = fig.add_subplot(3,2, 3)
ax3.title.set_text('sharpen')
plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))

img4 = preprocess.sharpen2(img)
ax4 = fig.add_subplot(3,2, 4)
ax4.title.set_text('sharpen2')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

img5 = preprocess.denoise(img4)
ax5 = fig.add_subplot(3,2, 5)
ax5.title.set_text('denoise')
plt.imshow(cv2.cvtColor(img5, cv2.COLOR_BGR2RGB))


plt.show()