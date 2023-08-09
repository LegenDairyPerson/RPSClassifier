import cv2 as cv
import os
from PIL import Image
import numpy
def count_in_dir(dir_path):
    n = 0
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            n += 1
    return n
def loop_list(list, current):
    if current < len(list)-1:
        return list[current+1]
    elif current == len(list)-1:
        return list[0]

modes = ["train", "test"]
types = ["paper", "rock", "scissors"]

mode = modes[0]
type = types[0]
auto = False

camera = cv.VideoCapture(0)

while True:
    dir = "custom-test-set" if mode == "test" else "custom-rps"
    dir_path = rf'{dir}\{type}'
    count = count_in_dir(dir_path)
    ret, frame = camera.read()

    if not ret:
        continue

    frame = cv.resize(frame, (300, 300), interpolation=cv.INTER_AREA)
    img = Image.fromarray(frame)
    frame = cv.putText(frame, f"Auto: {auto}", (0, 225), cv.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
    frame = cv.putText(frame, f"Mode: {mode}", (0, 250), cv.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
    frame = cv.putText(frame, f"type: {type}", (0, 275), cv.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
    frame = cv.putText(frame, f"Count: {str(count)}", (0, 300), cv.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
    cv.imshow("cam feed", frame)

    key = cv.waitKey(1) & 0xff
    if key == ord("q"):
        break
    elif (key == ord("s")) and (not auto):
        img.save(f"{dir}\{type}\{type}{count}.png")
    elif key == ord("m"):
        mode = loop_list(modes, modes.index(mode))
    elif key == ord("t"):
        type = loop_list(types, types.index(type))
    elif key == ord("a"):
        auto = not auto

    if auto:
        img.save(f"{dir}\{type}\{type}{count}.png")

camera.release()
cv.destroyAllWindows()
