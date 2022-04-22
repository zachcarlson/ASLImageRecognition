import cv2
import os
current_letter = 'a'

dir_name = os.path.dirname(__file__)
print(dir_name)

filename = os.path.join(dir_name, 'Videos', 'A.mp4')
print(filename)

frame_location = os.path.join(dir_name, 'framed_videos', 'letter_' + current_letter)
print(frame_location)

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