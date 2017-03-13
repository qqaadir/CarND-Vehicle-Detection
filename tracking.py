import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from numpy.linalg import inv
from skimage import feature
import copy

#
# Tracker class, with functions for tracking objects
#
class Tracker():
    def __init__(self):
        self.predictions = []

    def get_pred_coords(self):
        coords = []
        for x in self.predictions:
            coords.append([x.state[0],x.state[1]])
        return coords


    def associate(self,measurements):
        cost = np.zeros((len(self.predictions),len(measurements)))
        coords = self.get_pred_coords()

        for i,x in enumerate(coords):
            diff = np.subtract(np.transpose(x),measurements)
            if(len(measurements) == 1):
                cost[i] = np.sqrt(np.sum(np.power(diff,2)))
            else:
                cost[i,:] = np.sqrt(np.sum(np.power(diff,2),axis=1))

        
        row_ind, col_ind = linear_sum_assignment(cost)
        costs = cost[row_ind,col_ind]
        to_remove_r = []
        to_remove_c = []
        for i,c in enumerate(costs):
            if(c>7.5):
                to_remove_r = row_ind[i]
                to_remove_c = col_ind[i]
        
        #np.delete(row_ind, to_remove_r, None)
        #np.delete(col_ind, to_remove_c, None)
        return row_ind,col_ind

    def get_model_image(self,image):
        hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        h = hsv[:,:,0]
        s = hsv[:,:,1]
        lbp = self.get_lbp_image(image)

        roi_image = cv2.merge((h,s,lbp.astype('uint8')))

        return roi_image

    def get_model_histogram(self,image, roi):
        x = roi[0][0]
        y = roi[0][1]
        width = roi[1][0] - roi[0][0]
        height = roi[1][1] - roi[0][1]

        hslbp = self.get_model_image(image)
        roi_image = hslbp[y:y+height,x:x+width,:]
        #H, edges = np.histogramdd([h.ravel(),s.ravel(),lbp.ravel()], bins = (48, 48, 27), normed=True)
        #hist = cv2.calcHist(roi_image, [0, 1, 2], None, [48, 48,27], [0, 180, 0, 256, 0, 26])
        hist = cv2.calcHist(roi_image, [0, 1], None, [48,48], [0, 180, 0, 256])
        cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)

        return hist

    def get_lbp_image(self,image):
        numPoints = 24
        radius = 8
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        lbp = feature.local_binary_pattern(gray, numPoints, radius, method="uniform")
        return lbp

    def camshift_tracking(self,image, bbox, roi_hist):
        x = bbox[0][0]
        y = bbox[0][1]
        w = bbox[1][0] - bbox[0][0]
        h = bbox[1][1] - bbox[0][1]

        track_window = (x,y,w,h)

        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        img = self.get_model_image(image)
        #dst = cv2.calcBackProject([img],[0,1,2],roi_hist,[0,180,0,256,0,26],1)
        dst = cv2.calcBackProject([img],[0,1],roi_hist,[0,180,0,256],1)
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        x,y,w,h = track_window

        return [(x,y), (x+w,y+h)]


#
# TrackedObject class, for storing tracked car state info and Kalman filter variables
#
class TrackedObject:
    def __init__(self,center_x,center_y, histogramModel):
        self.init_kalman_filter()
        self.counter = 0
        self.state = np.array([center_x,center_y,0,0]) # assume initial velocity is zero
        self.state = self.state.reshape(4,1)
        self.bboxes = []
        self.histogramModel = histogramModel
        self.n = 4
        

    def init_kalman_filter(self):
        #state 4 values:(x,y,v_x, v_y)
        #measurement 2 values:(x,y)
        
        self.H = np.array([[1,0,0,0],[0,1,0,0]],np.float32)# Measurement function
        self.P = np.array([[0,0,0,0],[0,0,0,0],[0,0,1000,0],[0,0,0,1000]],np.float32) #uncertainty covariance
        self.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)# State transition function
        self.R = np.array([[10,0],[0,10]],np.float32) #measurement uncertainty
        self.I = np.identity(4)


    def predict(self):
        # prediction
        self.state = np.matmul(self.F,self.state) 
        self.P = np.matmul(np.matmul(self.F,self.P),np.transpose(self.F))
        return self.state

    def measurement(self,z):
        y = z - np.matmul(self.H,self.state)
        S = np.matmul(np.matmul(self.H,self.P),self.H.transpose()) + self.R
        K = np.matmul(np.matmul(self.P,np.transpose(self.H)),inv(S))
        self.state = self.state + np.matmul(K,y)
        self.P = np.matmul((self.I - np.matmul(K,self.H)),self.P)
        return self.state

    def add_bbox(self,bbox):
        if(len(self.bboxes) > self.n):
            self.bboxes.pop(0)
        self.bboxes.append(bbox)

    def average_bbox(self):
        avg_width = 0
        avg_height = 0
        for bbox in self.bboxes:
            avg_width += bbox[1][0] - bbox[0][0]
            avg_height += bbox[1][1] - bbox[0][1]

        avg_height = avg_height/len(self.bboxes)
        avg_width = avg_width/len(self.bboxes)

        x = int(self.state[0][0] - avg_width/2)
        y = int(self.state[1][0] - avg_height/2)

        return [(x,y),(x+int(avg_width),y+int(avg_height))]

    def bbox_to_roi(self):
        bbox = self.average_bbox()
        x = bbox[0][0]
        y = bbox[0][1]
        w = bbox[1][0] - bbox[0][0]
        h = bbox[1][1] - bbox[0][1]

        return (x,y,w,h)
