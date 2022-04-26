import cv2
import time
import numpy as np
import pandas as pd
import os
from scipy import signal
from Mode import  *

#Set spatial size that CNN expects
spatial_size = (368,368)
threshold = 0.1

#Read the model weights into memory
net = cv2.dnn.readNetFromCaffe(protoFile,weightsFile)


class poseDetector():

    '''
    Detect keypoints in a passed frame
    '''
    def find_pose(self,img):
        self.img = img
        self.t = time.time()

        #Get a blob from the frame
        self.inputBlob = cv2.dnn.blobFromImage(self.img, 1.0 / 255, spatial_size , (0, 0, 0), swapRB=False, crop=False)

        # set the input and perform a forward pass
        net.setInput(self.inputBlob)
        self.output = net.forward()

        #get the output shape
        self.output_width,self.output_height = self.output.shape[2],self.output.shape[3]

        # Empty list to store the detected keypoints
        self.points = []

        self.detect_points()


    def detect_points(self):
        self.img_copy = np.copy(self.img)
        img_width,img_height = self.img.shape[1], self.img.shape[0]
        radius,circle_color,line_color = 6,(0, 255, 255),(0,0,255)

        for i in range(nPoints):

            # find probability that point is correct
            _, probability, _, point = cv2.minMaxLoc(self.output[0, i, :, :])

            # Scale the point to fit on the original image
            x, y = (img_width * point[0]) / self.output_width, (img_height * point[1]) / self.output_height
            center_xy = tuple(np.array([x, y], int))

            # Is the point likely to be correct?
            if probability > threshold:
                cv2.circle(self.img_copy, center_xy , 6,circle_color , thickness=-1)
                cv2.putText(self.img_copy, "{}".format(i), center_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.5,line_color , 1)

                # Add the point to the list if the probability is greater than the threshold
                self.points.append(center_xy)
            else:
                self.points.append(None)


    '''
    Draw points on to the frame
    '''
    def draw_points(self):
        return self.img_copy

    '''
    Draw the skeleton from the detected keypoints
    '''
    def draw_skeleton(self):
        radius, circle_color, line_color = 6, (255, 255, 255), (0, 0, 255)
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if self.points[partA] and self.points[partB]:
                cv2.line(self.img, self.points[partA], self.points[partB], line_color, 3, lineType=cv2.LINE_AA)
                cv2.circle(self.img, self.points[partA], radius, circle_color, thickness=-1, lineType=cv2.FILLED)
                cv2.circle(self.img, self.points[partB], radius, circle_color, thickness=-1, lineType=cv2.FILLED)

        cv2.putText(self.img, "time taken = {:.2f} sec".format(time.time() - self.t), (10,20), cv2.FONT_HERSHEY_COMPLEX, .8,
                    (150,0,0), 1, lineType=cv2.LINE_AA)

        return self.img


class poseSmoother():

    def __init__(self,input_src):
        self.input_src = input_src
        self.filename = os.path.splitext(input_src)[0]

    '''
    Save the detected keypoints in csv file for post processing
    '''
    def save_posedata(self):
        data = []
        prev_x, prev_y = [0]*nPoints, [0]*nPoints
        radius , circle_color = 6,(0,255,255)
        cap = cv2.VideoCapture(self.input_src)
        while True:

            succ, img = cap.read()
            if not succ: break

            # get the image shape
            img_width, img_height = img.shape[1], img.shape[0]

            # get a blob from the image
            inputBlob = cv2.dnn.blobFromImage(img, 1.0 / 255, spatial_size, (0, 0, 0), swapRB=False, crop=False)

            # set the input and perform a forward pass
            net.setInput(inputBlob)
            output = net.forward()

            # get the output shape
            output_width, output_height = output.shape[2], output.shape[3]

            # Empty list to store the detected keypoints
            x_data, y_data = [], []

            # Iterate through the body parts
            for i in range(nPoints):

                # find probability that point is correct
                _, probability, _, point = cv2.minMaxLoc(output[0, i, :, :])

                # Scale the point to fit on the original image
                x, y = (img_width * point[0]) / output_width, (img_height * point[1]) / output_height

                # Is the point likely to be correct?
                if probability > threshold:
                    x_data.append(x), y_data.append(y)
                    xy = tuple(np.array([x, y], int))
                    cv2.circle(img, xy,radius,circle_color, -1)

                # No? use the location in the previous frame
                else:
                    x_data.append(prev_x[i])
                    y_data.append(prev_y[i])


            # add these points to the list of data
            data.append(x_data + y_data)
            prev_x, prev_y = x_data, y_data

            cv2.imshow('img', img)
            k = cv2.waitKey(1)
            if k == 27: break

        cv2.destroyAllWindows()
        df = pd.DataFrame(data)
        df.to_csv(self.filename+".csv", index=False)
        print('save complete')


    '''
    Apply the smoothing filter on the keypoints detected saved in the csv file
    '''
    def smooth_pose(self):
        df = pd.read_csv(self.filename+".csv")

        radius, circle_color, line_color = 6, (255, 255, 255), (0, 0, 255)
        window_length, polyorder = 13,2

        cap = cv2.VideoCapture(self.input_src)
        succ, img = cap.read()
        vid_writer = cv2.VideoWriter(self.filename+'_savgol_filter(x'+str(window_length)+","+str(polyorder)+').avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                     (img.shape[1], img.shape[0]))

        # Smooth it out
        for i in range(df.shape[1]):
            df[str(i)] = signal.savgol_filter(df[str(i)], window_length, polyorder)

        frame_number = 0
        while True:

            succ, img = cap.read()
            if not succ: break

            values = np.array(df.values[frame_number], int)

            points = []
            points = list(zip(values[:nPoints], values[nPoints:]))

            for point in points:
                xy = tuple(np.array([point[0], point[1]], int))
                cv2.circle(img, xy, radius,circle_color, -1)

            # Draw Skeleton
            for pair in POSE_PAIRS:
                partA = pair[0]
                partB = pair[1]
                cv2.line(img, points[partA], points[partB], line_color, 3, lineType=cv2.LINE_AA)

            cv2.putText(img, "Savgol_filter(x,"+str(window_length)+","+str(polyorder)+")", (10, 20), cv2.FONT_HERSHEY_COMPLEX, .8,(150,0,0), 1, lineType=cv2.LINE_AA)
            cv2.imshow('Skeleton', img)
            k = cv2.waitKey(100)
            if k == 27: break
            vid_writer.write(img)
            frame_number += 1

        vid_writer.release()
        cv2.destroyAllWindows()




