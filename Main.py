import cv2
import os
import PoseModule as pm
import mimetypes


def image_detector(input_source):
    detector = pm.poseDetector()
    filename = os.path.splitext(input_source)[0]
    img = cv2.imread(input_source)
    detector.find_pose(img)
    keypoints = detector.draw_points()
    skeleton = detector.draw_skeleton()
    cv2.imshow("keypoints", keypoints)
    cv2.imshow("skeleton", skeleton)
    cv2.imwrite(filename + "_keypoints.jpg", keypoints)
    cv2.imwrite(filename + "_skeleton.jpg", skeleton)

def video_detector(input_source):
    detector = pm.poseDetector()
    filename = os.path.splitext(input_source)[0]
    cap = cv2.VideoCapture(input_source)
    succ, frame = cap.read()
    vid_writer = cv2.VideoWriter(filename + ".avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                 (frame.shape[1], frame.shape[0]))

    while True:
        succ, frame = cap.read()
        if not succ: break
        detector.find_pose(frame)
        skeleton = detector.draw_skeleton()
        cv2.imshow("skeleton", skeleton)
        vid_writer.write(skeleton)
        k = cv2.waitKey(100)
        if k == 27: break

    vid_writer.release()
    cv2.destroyAllWindows()

def video_smoother(input_source):
    smoother = pm.poseSmoother(input_source)
    smoother.save_posedata()
    smoother.smooth_pose()


def main():

    # Input can be image or video
    input_source = "mediafiles/exercise/exercise.mp4"

    # Extract filetype
    filetype = mimetypes.guess_type(input_source)[0]

    if filetype != None:
        mediatype = filetype.split('/')[0]
        if mediatype == 'video':
            video_detector(input_source)
            # Smoothes the poses and saves in another file
            video_smoother(input_source)

        elif mediatype == 'image':
            image_detector(input_source)

        else:
            print("File type not supported")
    else:
        print("Not a valid file format")

if __name__ == '__main__':
    main()