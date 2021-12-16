import cv2, sys, os
import numpy as np
import json
import glob

class Detector:
	# This example of a detector detects faces. However, you have annotations for ears!

	#cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'lbpcascade_frontalface.xml'))
	cascade_l_ear = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cascades/haarcascade_mcs_leftear.xml"))
	cascade_r_ear = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cascades/haarcascade_mcs_rightear.xml"))

	def detect(self, img, scale_factor, min_neighbours):
		det_list_l = self.cascade_l_ear.detectMultiScale(img, scale_factor, min_neighbours)
		det_list_r = self.cascade_r_ear.detectMultiScale(img, scale_factor, min_neighbours)


		# When no results 	
		if  type(det_list_l) is not np.ndarray or det_list_l.size == 0:
			return det_list_r
		if type(det_list_r) is not np.ndarray or det_list_r.size == 0:
			return det_list_l


		return np.block([[det_list_l], [det_list_r]])

if __name__ == '__main__':
	# Set how many images we will run this through
	n_images = 50
	# Read the config and get images path
	with open('../../config.json') as config_file:
		config = json.load(config_file)
	images_path = "../../"+config['images_path']
	
	# Sort images, only get the png's (non detected)
	im_list = sorted(glob.glob(images_path + '/*.png', recursive=True))

	for ix, im_name in enumerate(im_list):
		# Show progress
		
		if ix > n_images:
			break

		# Read an image
		img = cv2.imread(im_name)

		detector = Detector()
		detected_loc = detector.detect(img)
		#print(detected_loc)
		for x, y, w, h in detected_loc:
			cv2.rectangle(img, (x,y), (x+w, y+h), (128, 255, 0), 4)
		print("writing :" + im_name + '.detected.jpg\n')
		cv2.imwrite(im_name + '.detected.jpg', img)
