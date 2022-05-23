import cv2
import os
import sys

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

for x in letters:

    current_letter = x

    dir_name = os.path.dirname(__file__)

    filename = os.path.join(dir_name, 'Videos', current_letter + '.mp4')

    frame_location = os.path.join(dir_name, 'framed_videos', 'letter_' + current_letter)
    CHECK_FOLDER = os.path.isdir(frame_location)

    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(frame_location)

    capture = cv2.VideoCapture(filename)

    frameNr = 0

    while (True):

        success, frame = capture.read()

        if success:
            cv2.imwrite(f'{frame_location}/{current_letter}_frame_{frameNr}.jpg', frame)

        else:
            break

        frameNr = frameNr + 1
        print(frameNr)

    capture.release()