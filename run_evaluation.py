import cv2
import numpy as np
import glob
import os
import time 
from pathlib import Path
import json
from preprocessing.preprocess import Preprocess
from metrics.evaluation import Evaluation
import matplotlib.pyplot as plt

class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config.json') as config_file:
            config = json.load(config_file)

        self.images_path = config['images_path']
        self.annotations_path = config['annotations_path']

    def get_annotations(self):
        # Loop over all files
        file_names = sorted(next(os.walk(self.annotations_path), (None, None, []))[2])  # [] if no file

        annot_list = []
        for file in file_names:
            with open(self.annotations_path+ "/" +file) as f:
                lines = f.readlines()
                annot = []
                for line in lines:
                    l_arr = line.split(" ")[1:5]
                    l_arr = [int(i) for i in l_arr]
                    annot.append(l_arr)
                annot_list.append(annot)

        return annot_list

    def normalizedBB_to_absoluteBB(self, detected_list):
        new_list = []
        for detected_for_img in detected_list:
            det_image = []
            for bb in detected_for_img:
                x, y, w, h = bb
                det_image.append([int(x - w/2), int(y - h/2), int(w), int(h)])

            new_list.append(det_image)

        return new_list

    def get_avg_iou(self, prediction_list, annot_list):
        eval = Evaluation()
        iou_arr = []
        for ix, prediction in enumerate(prediction_list):
            p, gt = eval.prepare_for_detection(prediction, annot_list[ix])
            iou = eval.iou_compute(p, gt)
            iou_arr.append(iou)
        return np.average(iou_arr)

    # Get abs. difference between the ammount of annotated vs predicted ears
    def get_avg_annot_detect_diff(self, prediction_list, annot_list):
        eval = Evaluation()
        diff_arr = []
        for ix, prediction in enumerate(prediction_list):
            diff_arr.append(abs(len(annot_list[ix]) - len(prediction)))
        return np.average(diff_arr)

    # Plot a simple bar graph
    def simple_graph(self, x, y, x_label, y_label, title, file_name):
        plt.bar(x, y, width = 0.8, color = ['blue'])
        # naming the x-axis
        plt.xlabel(x_label)
        # naming the y-axis
        plt.ylabel(y_label)
        # plot title
        plt.title(title)
        plt.savefig(file_name + ".png")

    def run_evaluation(self, annot_list):
        acc_time_arr = []

        im_list = sorted(glob.glob(self.images_path + '/*.png', recursive=True))

        preprocess = Preprocess()

        import detectors.cascade_detector.detector as cascade_detector
        cascade_detector = cascade_detector.Detector()

        import detectors.your_super_detector.detect as yolo_detector
        yolov5_detector = yolo_detector.DetectorYOLOv5()

    


        #=================== YOLOv5 different epochs ===================
        epochs = ["10", "25", "60", "100"]
        for epoch in epochs:
            start_time = time.time()
            print("\n--- Running YOLOv5 (" + epoch + " epochs) ---")
            prediction_list = yolov5_detector.run("detectors/your_super_detector/ears_" + epoch + "ep.pt", "data/ears/test")
            run_time = time.time() - start_time
            prediction_list = ev.normalizedBB_to_absoluteBB(prediction_list)
            print("\n--- %.2f seconds ---" % (run_time))
            # Calculate IoU 
            avg_iou = ev.get_avg_iou(prediction_list, annot_list)
            avg_diff = ev.get_avg_annot_detect_diff(prediction_list, annot_list)
            print("\nAverage IOU:", f"{avg_iou:.2%}")
            print("\nAverage DIFF:", f"{avg_diff}")
            print("\n")
            #Save results
            acc_time_arr.append(["YOLOv5 " + epoch + " epoch", f"{avg_iou:.3}", f"{avg_diff:.3}", f"{run_time:.2f}"])
            
        print(acc_time_arr)

        #acc_time_arr = [['YOLOv5 10 epoch', '0.0355', '0.336', '9.09'], ['YOLOv5 25 epoch', '0.863', '0.1', '5.72'], ['YOLOv5 60 epoch', '0.891', '0.076', '5.68'], ['YOLOv5 100 epoch', '0.896', '0.068', '5.73']]

        yolo_avg_iou_arr = []
        for val in acc_time_arr:
            yolo_avg_iou_arr.append(float(val[1]))

        ev.simple_graph(epochs, yolo_avg_iou_arr, "Epochs", "average IoU", "Training epochs IoU correlation", "Training_epochs_IoU_correlation")
        #===================/ YOLOv5 different epochs ===================

        #=================== VJ scale factor comparison ===================
        acc_time_arr_vj_scale = []
        scale_factors = [1.05, 1.075, 1.1, 1.5, 2]
        for scale_factor in scale_factors:
            start_time = time.time()
            prediction_list = []
            print("\n--- Running VJ (Scale factor " + str(scale_factor) + ") ---")
            for ix, im_name in enumerate(im_list):
                img = cv2.imread(im_name)
                np_list = cascade_detector.detect(img, scale_factor, 1)
                if type(np_list) is not tuple:
                    prediction_list.append(np_list.tolist())
                else:
                    prediction_list.append([])
                
            run_time = time.time() - start_time
            print("\n--- %.2f seconds ---" % (run_time))
            # Calculate IoU 
            avg_iou = ev.get_avg_iou(prediction_list, annot_list)
            print("\nAverage IOU:", f"{avg_iou:.2%}")
            print("\n")
            #Save results
            acc_time_arr_vj_scale.append(["Viola-Jones(scale factor" + str(scale_factor) + ")", f"{avg_iou:.3}", f"{run_time:.2f}"])
        
        print(acc_time_arr_vj_scale)
        
        #scale_factors = [1.05, 1.075, 1.1, 1.5, 2]
        #acc_time_arr_vj_scale = [['Viola-Jones(scale factor1.05)', '0.303', '56.77'], ['Viola-Jones(scale factor1.075)', '0.303', '39.67'], ['Viola-Jones(scale factor1.1)', '0.275', '30.36'], ['Viola-Jones(scale factor1.5)', '0.163', '9.66'], ['Viola-Jones(scale factor2)', '0.129', '7.81']]
        
        vj_scale_avg_iou_arr = []
        vj_scale_time = []
        for val in acc_time_arr_vj_scale:
            vj_scale_time.append(float(val[2]))
            vj_scale_avg_iou_arr.append(float(val[1]))

        str_scale_factors = list(map(str, scale_factors))

        ev.simple_graph(str_scale_factors, vj_scale_avg_iou_arr, "Scale factor", "average IoU", "VJ scale factor IoU correlation", "vj_scale_iou_corr")
        ev.simple_graph(str_scale_factors, vj_scale_time, "Scale factor", "Total time", "VJ scale factor total time", "vj_scale_time_corr")
        #===================/ VJ scale factor comparison ===================

        #=================== VJ max neighbours comparison ===================
        scale_factor = 1.075
        acc_time_arr_vj_neighbors = []
        max_neighbours = [0, 1, 2, 3, 4, 5]
        for max_neighbour in max_neighbours:
            start_time = time.time()
            prediction_list = []
            print("\n--- Running VJ (Scale factor: " + str(scale_factor) + ", Max-neighbours " + str(max_neighbour) + ") ---")
            for ix, im_name in enumerate(im_list):
                img = cv2.imread(im_name)
                np_list = cascade_detector.detect(img, scale_factor, max_neighbour)
                if type(np_list) is not tuple:
                    prediction_list.append(np_list.tolist())
                else:
                    prediction_list.append([])
                
            run_time = time.time() - start_time
            print("\n--- %.2f seconds ---" % (run_time))
            # Calculate IoU 
            avg_iou = ev.get_avg_iou(prediction_list, annot_list)
            avg_diff = ev.get_avg_annot_detect_diff(prediction_list, annot_list)
            print("\nAverage IOU:", f"{avg_iou:.2%}")
            print("\nAverage DIFF:", f"{avg_diff}")
            print("\n")
            #Save results
            acc_time_arr_vj_neighbors.append(["Viola-Jones(Scale factor: " + str(scale_factor) + ", Max-neighbours  " + str(max_neighbour) + ")", f"{avg_iou:.3}", f"{avg_diff:.3}", f"{run_time:.2f}"])
        print(acc_time_arr_vj_neighbors)

        #acc_time_arr_vj_neighbors = [['Viola-Jones(Scale factor: 1.075, Max-neighbours  0)', '0.304', '7.05', '39.51'], ['Viola-Jones(Scale factor: 1.075, Max-neighbours  1)', '0.303', '0.716', '39.41'], ['Viola-Jones(Scale factor: 1.075, Max-neighbours  2)', '0.272', '0.74', '39.42'], ['Viola-Jones(Scale factor: 1.075, Max-neighbours  3)', '0.254', '0.792', '39.53'], ['Viola-Jones(Scale factor: 1.075, Max-neighbours  4)', '0.229', '0.832', '39.84'], ['Viola-Jones(Scale factor: 1.075, Max-neighbours  5)', '0.209', '0.856', '39.55']]

        vj_neighbors_avg_iou_arr = []
        vj_neighbors_time = []
        vj_neighbors_diff = []
        for val in acc_time_arr_vj_neighbors:
            vj_neighbors_avg_iou_arr.append(float(val[1]))
            vj_neighbors_diff.append(float(val[2]))
            vj_neighbors_time.append(float(val[3]))

        str_max_neighbors = list(map(str, max_neighbours))

        ev.simple_graph(str_max_neighbors, vj_neighbors_avg_iou_arr, "Max neighbors", "average IoU", "VJ Max neighbors IoU correlation(1.075 scale-factor)", "vj_neighbors_iou_corr")
        ev.simple_graph(str_max_neighbors, vj_neighbors_diff, "Max neighbors", "Total time", "VJ Max neighbors average detection diff(1.075 scale-factor)", "vj_neighbors_diff")
        ev.simple_graph(str_max_neighbors, vj_neighbors_time, "Max neighbors", "Total time", "VJ Max neighbors total time(1.075 scale-factor)", "vj_neighbors_time_corr")
        #===================/ VJ max neighbours comparison ===================

        #=================== VJ preprocess comparison ===================
        scale_factor = 1.075
        max_neighbours = 1
        acc_time_arr_vj_preprocess = []
        preprocess_functions = [preprocess.histogram_equlization_rgb, preprocess.gaussian_threshold, preprocess.sharpen, preprocess.sharpen2, preprocess.denoise]
        for preprocess_f in preprocess_functions:
            start_time = time.time()
            prediction_list = []
            print("\n--- Running VJ (" + preprocess_f.__name__ + ") ---")
            for ix, im_name in enumerate(im_list):
                img = cv2.imread(im_name)
                img = preprocess_f(img)
                np_list = cascade_detector.detect(img, scale_factor, max_neighbours)
                if type(np_list) is not tuple:
                    prediction_list.append(np_list.tolist())
                else:
                    prediction_list.append([])
                
            run_time = time.time() - start_time
            print("\n--- %.2f seconds ---" % (run_time))
            # Calculate IoU 
            avg_iou = ev.get_avg_iou(prediction_list, annot_list)
            print("\nAverage IOU:", f"{avg_iou:.2%}")
            print("\n")
            #Save results
            acc_time_arr_vj_preprocess.append(["Viola-Jones(" + preprocess_f.__name__ + ")", f"{avg_iou:.3}", f"{run_time:.2f}"])
        print(acc_time_arr_vj_preprocess)

        #acc_time_arr_vj_preprocess = [['Viola-Jones(histogram_equlization_rgb)', '0.234', '46.77'], ['Viola-Jones(gaussian_threshold)', '0.0', '46.42'], ['Viola-Jones(sharpen)', '0.16', '40.34'], ['Viola-Jones(sharpen2)', '0.187', '53.31'], ['Viola-Jones(denoise)', '0.275', '427.08']]

        vj_preprocess_avg_iou = []
        vj_preprocess_time = []
        for val in acc_time_arr_vj_preprocess:
            vj_preprocess_avg_iou.append(float(val[1]))
            vj_preprocess_time.append(float(val[2]))

        str_function_names = []
        for fun in preprocess_functions:
            str_function_names.append(fun.__name__)

        ev.simple_graph(str_function_names, vj_preprocess_avg_iou, "Preprocess function", "average IoU", "VJ preprocess avg IoU(1.075 scale-factor, 1 max-neighbours)", "vj_preprocess_iou_corr")
        ev.simple_graph(str_function_names, vj_preprocess_time, "Preprocess function", "Total time", "VJ preprocess functions total time comparison(1.075 scale-factor, 1 max-neighbours)", "vj_preprocess_total_time")
        #===================/ VJ preprocess comparison ===================



        #=================== best vj vs 100ep YOLOv5 ===================
        #acc_time_arr = [['YOLOv5 10 epoch', '0.0355', '0.336', '9.09'], ['YOLOv5 25 epoch', '0.863', '0.1', '5.72'], ['YOLOv5 60 epoch', '0.891', '0.076', '5.68'], ['YOLOv5 100 epoch', '0.896', '0.068', '5.73']]
        #acc_time_arr_vj_neighbors = [['Viola-Jones(Scale factor: 1.075, Max-neighbours  0)', '0.304', '7.05', '39.51'], ['Viola-Jones(Scale factor: 1.075, Max-neighbours  1)', '0.303', '0.716', '39.41'], ['Viola-Jones(Scale factor: 1.075, Max-neighbours  2)', '0.272', '0.74', '39.42'], ['Viola-Jones(Scale factor: 1.075, Max-neighbours  3)', '0.254', '0.792', '39.53'], ['Viola-Jones(Scale factor: 1.075, Max-neighbours  4)', '0.229', '0.832', '39.84'], ['Viola-Jones(Scale factor: 1.075, Max-neighbours  5)', '0.209', '0.856', '39.55']]

        ev.simple_graph(["Viola-Jones", "YOLOv5"], [float(acc_time_arr_vj_neighbors[1][1]), float(acc_time_arr[3][1])], "Method", "average IoU", "YOLOv5(100ep) vs VJ (1.075 s-f, 1 max-n) avg IoU", "vj_vs_yolo_iou_corr")
        ev.simple_graph(["Viola-Jones", "YOLOv5"], [float(acc_time_arr_vj_neighbors[1][2]), float(acc_time_arr[3][2])], "Method", "average Diff", "YOLOv5(100ep) vs VJ (1.075 s-f, 1 max-n) avg Diff", "vj_vs_yolo_diff")
        ev.simple_graph(["Viola-Jones", "YOLOv5"], [float(acc_time_arr_vj_neighbors[1][3]), float(acc_time_arr[3][3])], "Method", "Total time", "YOLOv5(100ep) vs VJ (1.075 s-f, 1 max-n) total time", "vj_vs_yolo_total_time")
        #===================/ best vj vs 100ep YOLOv5 ===================



if __name__ == '__main__':
    ev = EvaluateAll()
    annot_list = ev.get_annotations()
    ev.run_evaluation(annot_list)