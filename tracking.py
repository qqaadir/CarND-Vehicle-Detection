import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from numpy.linalg import inv
from skimage import feature
import copy

class Tracker():
    def __init__(self):
        #state 4 values:(x,y,v_x, v_y)
        #measurement 2 values:(x,y)
        self.kalman_filters = []
        self.predictions = []
    
    def get_bbox_centers(self,bboxes):
        centers = []
        for bbox in bboxes:
            centers.append((int((bbox[1][0] - bbox[0][0])/2 + bbox[0][0]),int((bbox[1][1] - bbox[0][1])/2 + bbox[0][1])))

        return centers

    def track(self,img,bboxes = None):
        measurements = []
        if(bboxes is not None):
            measurements = self.get_bbox_centers(bboxes)
            #print(len(self.predictions))
            if(len(self.predictions) == 0):
                #not tracking anything
                for i,point in enumerate(measurements):
                    xtop = int(bboxes[i][0][0])
                    ytop = int(bboxes[i][0][1])
                    width = int(bboxes[i][1][0] - xtop)
                    height = int(bboxes[i][1][1] - ytop)
                    self.predictions.append(TrackedObject(point[0],point[1],xtop,ytop,width,height, self.get_model_histogram(img,bboxes[i])))

            for i,x in enumerate(self.predictions):
                #x.update_prediction()
                #print("Predict")
                x.predict()
        else:
            bboxes = []
            for i,x in enumerate(self.predictions):
                bboxes.append(self.camshift_tracking(img,x.track_window,x.histogramModel))
                #x.update_prediction()
                #print("Predict")
                x.predict()
            measurements = self.get_bbox_centers(bboxes)

        cost = np.zeros((len(self.predictions),len(measurements)))
        coords = self.get_pred_coords()
        tracked_bboxes = []
        if(len(coords) == 1):
            val = measurements[0]
            self.predictions[0].add_bbox(bboxes[0])
            self.predictions[0].measurement(np.array([[np.float32(val[0])],[np.float32(val[1])]]))
            tracked_bboxes.append(self.predictions[0].average_bbox())
        else:
            for i,x in enumerate(coords):
                diff = np.subtract(x,measurements)
                cost[i,:] = np.sqrt(np.sum(np.power(diff,2),axis=1))

            row_ind, col_ind = linear_sum_assignment(cost)

            
            for i,x in enumerate(row_ind):
                val = measurements[col_ind[i]]
                #self.predictions[x].kalman.correct(np.array([[np.float32(val[0])],[np.float32(val[1])]]))

                self.predictions[x].add_bbox(bboxes[x])

                #print("Z:", val)
                #print("Measurement")
                self.predictions[x].measurement(np.array([[np.float32(val[0])],[np.float32(val[1])]]))
                #self.predictions[x].update_prediction()
                tracked_bboxes.append(self.predictions[x].average_bbox())
                

        return tracked_bboxes

    def no_detections(self):
        for x in self.predictions:
            x.counter += 1

    def get_pred_coords(self):
        coords = []
        for x in self.predictions:
            coords.append([x.state[0],x.state[1]])
        return coords

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
        hist = cv2.calcHist(roi_image, [0, 1, 2], None, [48, 48,27], [0, 180, 0, 256, 0, 26])
        cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)

        return hist

    def get_lbp_image(self,image):
        numPoints = 24
        radius = 8
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        lbp = feature.local_binary_pattern(gray, numPoints, radius, method="uniform")
        return lbp

    def camshift_tracking(self,image, track_window, roi_hist):
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        img = self.get_model_image(image)
        dst = cv2.calcBackProject([img],[0,1,2],roi_hist,[0,180,0,256,0,26],1)
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        x,y,w,h = track_window

        #out_img = copy.copy(image)
        #cv2.rectangle(out_img, (x,y), (x+w,y+h), (0,0,255), 6)
        #cv2.imshow("cam",out_img)
        #cv2.waitKey(0)

        return [(x,y), (x+w,y+h)]



class TrackedObject:
    def __init__(self,center_x,center_y,xtop,ytop,w,h, histogramModel):
        self.init_kalman_filter()
        self.counter = 0
        self.state = np.array([center_x,center_y,0,0]) # assume initial velocity is zero
        self.state = self.state.reshape(4,1)
        self.bboxes = []
        self.histogramModel = histogramModel
        self.n = 2
        self.track_window = (xtop,ytop,w,h)

    def init_kalman_filter(self):
        # Measurement model
        self.H = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        # Covariance
        self.P = np.array([[0,0,0,0],[0,0,0,0],[0,0,1000,0],[0,0,0,1000]],np.float32)
        # Motion model
        self.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        # Measurement covariance
        self.R = np.array([[10,0],[0,10]],np.float32)
        self.I = np.identity(4)


    def predict(self):
        # prediction
        self.state = np.matmul(self.F,self.state) #+ u
        self.P = np.matmul(np.matmul(self.F,self.P),np.transpose(self.F))
        return self.state

    def measurement(self,z):
        #print("Before")
        #print(self.state)
        y = z - np.matmul(self.H,self.state)
        S = np.matmul(np.matmul(self.H,self.P),self.H.transpose()) + self.R
        K = np.matmul(np.matmul(self.P,np.transpose(self.H)),inv(S))
        self.state = self.state + np.matmul(K,y)
        self.P = np.matmul((self.I - np.matmul(K,self.H)),self.P)
        #print("After")
        #print(self.state)
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

        self.track_window = (x,y,int(avg_width),int(avg_height))

        return [(x,y),(x+int(avg_width),y+int(avg_height))]
