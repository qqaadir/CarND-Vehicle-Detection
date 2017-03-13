import numpy as np
import cv2
from features import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import glob
from sklearn.svm import LinearSVC
import time
from sklearn.externals import joblib
from tracking import *
from detector import *
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
import copy

#dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
#svc = dist_pickle["svc"]


class FindCar:

    
    def __init__(self):
        self.car_tracker = Tracker()
        self.update_detection = True
        self.car_detector = Detector() 
        self.tracking = False
        self.count = 0
        self.non_detect = 10

    def train(self):
        car,non_car = self.load_training_images_names()
        self.car_detector.train(car,non_car)
    
    def process(self,img):
        draw_img = np.copy(img)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        
        test_img = np.copy(img)
        tracked_bboxes = []
        if(self.update_detection):
            self.update_detection = False
            self.count = 0
            bboxes = self.car_detector.multi_scale_detection(img)
            for bbox in bboxes:
                point = (int((bbox[1][0] - bbox[0][0])/2 + bbox[0][0]),int((bbox[1][1] - bbox[0][1])/2 + bbox[0][1]))
                tracked_obj = TrackedObject(point[0],point[1], self.car_tracker.get_model_histogram(img, bbox))
                #tracked_obj = self.car_tracker.add_tracked_object(img,bbox)
                tracked_bbox = self.car_tracker.camshift_tracking(img, bbox, tracked_obj.histogramModel)
                tracked_obj.add_bbox(tracked_bbox)
                if(self.car_detector.car_classify(test_img,tracked_bbox) == 1): 
                   
                    tracked_bboxes.append(tracked_obj.average_bbox())
                    self.car_tracker.predictions.append(tracked_obj)
                    
                    self.tracking = True
                else:
                    print("Bad detection")
                    bboxes.remove(bbox)
                
        elif(len(self.car_tracker.predictions)>0 and self.tracking):
            for tracked_obj in self.car_tracker.predictions:
                tracked_bboxes.append(self.car_tracker.camshift_tracking(img, tracked_obj.average_bbox(), tracked_obj.histogramModel))
            self.count += 1
        else:
            self.count += 1
        
        if(self.count == self.non_detect):
            self.update_detection = True
            
        
        for bbox in tracked_bboxes:
            cv2.rectangle(draw_img, bbox[0], bbox[1], (0,0,255), 6)
            x = (bbox[1][0] - bbox[0][0]) /2 + bbox[0][0]
            y = (bbox[1][1] - bbox[0][1]) /2 + bbox[0][1]
            cv2.circle(draw_img,(int(x),int(y)), 6, (0,0,255), -1)


        cv2.imshow("out",draw_img)
        cv2.waitKey(0)
        return draw_img

       

    def load_training_images_names(self):
        vehicles = []
        non_vehicles = []
        vehicles.append(glob.glob('training_imgs/vehicles/vehicles/GTI_Far/*.png'))
        vehicles.append(glob.glob('training_imgs/vehicles/vehicles/GTI_Left/*.png'))
        vehicles.append(glob.glob('training_imgs/vehicles/vehicles/GTI_MiddleClose/*.png'))
        vehicles.append(glob.glob('training_imgs/vehicles/vehicles/GTI_Right/*.png'))
        vehicles.append(glob.glob('training_imgs/vehicles/vehicles/GTI_Right/*.png'))
        vehicles.append(glob.glob('training_imgs/vehicles/vehicles/KITTI_extracted/*.png'))

        non_vehicles.append(glob.glob('training_imgs/non-vehicles/non-vehicles/Extras/*.png'))
        non_vehicles.append(glob.glob('training_imgs/non-vehicles/non-vehicles/GTI/*.png'))

        vehicles_out = np.concatenate(vehicles)
        non_vehicles_out = np.concatenate(non_vehicles)

        return vehicles_out, non_vehicles_out

    
   
    """def process_image(self,img):
        draw_img = np.copy(img)
        
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        test_img = np.copy(img)
        tracked_bboxes = []
        if(self.update_detection):
            bboxes = self.multi_scale_detection(img)

            for bbox in bboxes:
                print(self.car_classify(test_img,bbox))
            #detections = self.tracker.get_bbox_centers(bboxes)
            if(len(bboxes) > 0):
                self.no_cars_count = 0
                tracked_bboxes = self.tracker.track(img,bboxes)
                self.update_detection = False
                self.tracking = True
                self.count = 0
            else:
                self.update_detection = False
                self.no_cars_count += 1

        elif(self.tracking):
            tracked_bboxes = self.tracker.track(img)
            self.count += 1
            if(self.count == 10):
                self.count = 0
                self.update_detection = True
                self.tracking = False
        else:
            self.no_cars_count += 1
            if(self.no_cars_count == 10):
                self.update_detection = True
                self.no_cars_count = 0


        for bbox in tracked_bboxes:
            feature = extract_feature(test_img,resize=(64,64),cspace='BGR2YCrCb',spatial_size=self.spatial_size,hist_bins=self.hist_bins,hist_range=self.hist_range,
                                            orient=self.orient,pix_per_cell=self.pix_per_cel,cell_per_block=self.cell_per_block, hog_channel=self.hog_channel)

            test_feature = self.X_scaler.transform(feature.reshape(1,-1))
            test_prediction = self.svc.predict(test_feature)
            
            if(test_prediction == 1):
                cv2.rectangle(draw_img, bbox[0], bbox[1], (0,0,255), 6)
                x = (bbox[1][0] - bbox[0][0]) /2 + bbox[0][0]
                y = (bbox[1][1] - bbox[0][1]) /2 + bbox[0][1]
            
                cv2.circle(draw_img,(int(x),int(y)), 6, (0,0,255), -1)
            

        if(len(tracked_bboxes) > 0):
            cv2.imshow("tracking",draw_img)
            cv2.waitKey(0)
        #img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        return draw_img"""


def test_images():
    files = glob.glob('test_images/*.jpg')
    for x in files:
        img = cv2.imread(x)

if __name__ == "__main__":
   
    find_cars = FindCar()
    #find_cars.train()
    #output_video = 'test1_video.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    for frame in clip1.iter_frames():
        find_cars.process(frame)

    
    #white_clip = clip1.fl_image(find_cars.process)
    #white_clip.write_videofile(output_video, audio=False)