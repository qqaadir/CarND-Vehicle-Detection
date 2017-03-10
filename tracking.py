import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from numpy.linalg import inv


class Tracker():
    def __init__(self):
        #state 4 values:(x,y,v_x, v_y)
        #measurement 2 values:(x,y)
        self.kalman_filters = []
        self.predictions = []
       

    def add_heat(self,heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap# Iterate through list of bboxes
        
    def apply_threshold(self,heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    def draw_labeled_bboxes(self,img, labels):
        # Iterate through all detected cars
        bboxes = []
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            bboxes.append(bbox)
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return img, bboxes

    def get_bbox_centers(self,bboxes):
        centers = []
        for bbox in bboxes:
            centers.append((int((bbox[1][0] - bbox[0][0])/2 + bbox[0][0]),int((bbox[1][1] - bbox[0][1])/2 + bbox[0][1])))

        return centers

    def predict(self, detections, bboxes):
        print(len(self.predictions))
        if(len(self.predictions) == 0):
            #not tracking anything
            print("here")
            for x,y in detections:
                self.predictions.append(TrackedObject(x,y))
        
        for i,x in enumerate(self.predictions):
            #x.update_prediction()
            #print("Predict")
            x.predict()

        cost = np.zeros((len(self.predictions),len(detections)))
        coords = self.get_pred_coords()
        for i,x in enumerate(coords):
            diff = np.subtract(x,detections)
            cost[i,:] = np.sqrt(np.sum(np.power(diff,2),axis=1))
        
        row_ind, col_ind = linear_sum_assignment(cost)

        for i,x in enumerate(row_ind):
            val = detections[col_ind[i]]
            #self.predictions[x].kalman.correct(np.array([[np.float32(val[0])],[np.float32(val[1])]]))
            self.predictions[x].bboxes.append(bboxes[x])
            #print("Z:", val)
            #print("Measurement")
            self.predictions[x].measurement(np.array([[np.float32(val[0])],[np.float32(val[1])]]))
            #self.predictions[x].update_prediction()


    def no_detections(self):
        for x in self.predictions:
            x.counter += 1
        
    def get_pred_coords(self):
        coords = []
        for x in self.predictions:
            coords.append([x.state[0],x.state[1]])
        return coords


class TrackedObject:
    def __init__(self,x,y):
        self.init_kalman_filter()
        self.counter = 0
        self.state = np.array([x,y,0,0]) # assume initial velocity is zero 
        self.state = self.state.reshape(4,1)
        self.bboxes = []

    def init_kalman_filter(self):
        self.H = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        self.P = np.array([[0,0,0,0],[0,0,0,0],[0,0,1000,0],[0,0,0,1000]],np.float32)
        self.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        self.R = np.array([[10,0],[0,10]],np.float32)
        self.I = np.identity(4)
    

    def predict(self):
        # prediction
        self.state = np.matmul(self.F,self.state) #+ u
        self.P = np.matmul(np.matmul(self.F,self.P),np.transpose(self.F))
        return self.state

    def measurement(self,z):
        print("Before")
        print(self.state)
        y = z - np.matmul(self.H,self.state)
        S = np.matmul(np.matmul(self.H,self.P),self.H.transpose()) + self.R
        K = np.matmul(np.matmul(self.P,np.transpose(self.H)),inv(S))
        self.state = self.state + np.matmul(K,y)
        self.P = np.matmul((self.I - np.matmul(K,self.H)),self.P)
        print("After")
        print(self.state)
        return self.state
        