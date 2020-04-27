"""Demo code shows how to estimate human head pose.
Currently, human face is detected by a detector from an OpenCV DNN module.
Then the face box is modified a little to suits the need of landmark
detection. The facial landmark detection is done by a custom Convolutional
Neural Network trained with TensorFlow. After that, head pose is estimated
by solving a PnP problem.
"""
from argparse import ArgumentParser
from multiprocessing import Process, Queue
from threading import Thread
from queue import Queue as Q

import cv2
import numpy as np

from mark_detector import MarkDetector
from os_detector import detect_os
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer

import pyautogui
import math
import speech_recognition as sr

print("OpenCV version: {}".format(cv2.__version__))

# multiprocessing may not work on Windows and macOS, check OS for safety.
detect_os()

CNN_INPUT_SIZE = 128


def get_face(detector, img_queue, box_queue):
    """Get face from image queue. This function is used for multiprocessing"""
    while True:
        image = img_queue.get()
        box = detector.extract_cnn_facebox(image)
        box_queue.put(box)


def get_voice_command(voice_command_queue):
    # Initialize the recognizer
    r = sr.Recognizer()

    # Loop infinitely for user to
    # speak

    while (1):

        # Exception handling to handle
        # exceptions at the runtime
        try:
            # use the microphone as source for input.
            with sr.Microphone() as source2:

                # wait for a second to let the recognizer
                # adjust the energy threshold based on
                # the surrounding noise level
                r.adjust_for_ambient_noise(source2, duration=0.2)

                # listens for the user's input
                audio2 = r.listen(source2)

                # Using ggogle to recognize audio
                myText = r.recognize_google(audio2)
                print(myText)

                for st in myText.lower().split(' '):
                    voice_command_queue.put(st)

        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))

        except sr.UnknownValueError:
            print("unknown error occured")


def main():
    """MAIN"""
    cv2.namedWindow("Test")  # Create a named window
    cv2.moveWindow("Test", 900, 600)  # Move it to (40,30)

    screenWidth, screenHeight = pyautogui.size()
    st = 'Last command'

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    _, sample_frame = cap.read()

    # Introduce mark_detector to detect landmarks.
    mark_detector = MarkDetector()

    # Setup process and queues for multiprocessing.
    img_queue = Queue()
    box_queue = Queue()
    img_queue.put(sample_frame)
    box_process = Process(target=get_face, args=(
        mark_detector, img_queue, box_queue,))
    box_process.start()

    # Setting up process for listening to audio commands
    voice_command_queue = Q()
    stt_process = Thread(target=get_voice_command, args=(voice_command_queue,))
    stt_process.setDaemon(True)
    stt_process.start()

    # Introduce pose estimator to solve pose. Get one frame to setup the
    # estimator according to the image size.
    height, width = sample_frame.shape[:2]
    pose_estimator = PoseEstimator(img_size=(height, width))

    # Introduce scalar stabilizers for pose.
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    tm = cv2.TickMeter()

    while True:
        # Read frame, crop it, flip it, suits your needs.
        frame_got, frame = cap.read()
        if frame_got is False:
            break

        # Crop it if frame is larger than expected.
        # frame = frame[0:480, 300:940]

        # If frame comes from webcam, flip it so it looks like a mirror.
        frame = cv2.flip(frame, 2)

        # Pose estimation by 3 steps:
        # 1. detect face;
        # 2. detect landmarks;
        # 3. estimate pose

        # Feed frame to image queue.
        img_queue.put(frame)

        # Get face from box queue.
        facebox = box_queue.get()

        if facebox is not None:
            # Detect landmarks from image of 128x128.
            face_img = frame[facebox[1]: facebox[3],
                       facebox[0]: facebox[2]]
            face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            tm.start()
            marks = mark_detector.detect_marks([face_img])
            tm.stop()

            # Convert the marks locations from local CNN to global image.
            marks *= (facebox[2] - facebox[0])
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]

            # Uncomment following line to show raw marks.
            # mark_detector.draw_marks(
            #     frame, marks, color=(0, 255, 0))

            # Uncomment following line to show facebox.
            # mark_detector.draw_box(frame, [facebox])

            # Try pose estimation with 68 points.
            pose = pose_estimator.solve_pose_by_68_points(marks)

            # Stabilize the pose.
            steady_pose = []
            pose_np = np.array(pose).flatten()
            for value, ps_stb in zip(pose_np, pose_stabilizers):
                ps_stb.update([value])
                steady_pose.append(ps_stb.state[0])
            steady_pose = np.reshape(steady_pose, (-1, 3))

            # Uncomment following line to draw pose annotation on frame.
            # pose_estimator.draw_annotation_box(
            #     frame, pose[0], pose[1], color=(255, 128, 128))

            # Uncomment following line to draw stabile pose annotation on frame.
            pose_estimator.draw_annotation_box(
                frame, steady_pose[0], steady_pose[1], color=(255, 128, 128))

            # Uncomment following line to draw head axes on frame.
            endpoints = pose_estimator.getEndPoints(frame, steady_pose[0], steady_pose[1])

            deltax = endpoints[1][0] - endpoints[0][0]
            deltay = endpoints[1][1] - endpoints[0][1]

            xpos = math.floor((deltax + 44) * screenWidth / 88)
            ypos = math.floor((deltay + 14) * screenHeight / 58)

            # print(xpos, ypos)

            pyautogui.moveTo(xpos, ypos)

            if not voice_command_queue.empty():

                command = voice_command_queue.get_nowait()
                if 'click' in command or 'select' in command:
                    pyautogui.click()
                    st = 'Click'
                elif 'double' in command or 'in' in command:
                    pyautogui.doubleClick()
                    st = 'Double Click'
                elif 'right' in command or 'menu' in command or 'light' in command:
                    pyautogui.rightClick()
                    st = 'Right Click'

                print(command)

        cv2.putText(frame, st, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 20, 255)
        scale_percent = 30
        # calculate the 50 percent of original dimensions
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)

        # dsize
        dsize = (width, height)

        # resize image
        output = cv2.resize(frame, dsize)

        cv2.moveWindow("Test", screenWidth - width, screenHeight - height)

        # Show preview.
        cv2.imshow("Test", output)
        if cv2.waitKey(10) == 27:
            break

    # Clean up the multiprocessing process.
    box_process.terminate()
    box_process.join()



if __name__ == '__main__':
    main()
