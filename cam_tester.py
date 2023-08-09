import cv2 as cv
import main
import torch
test_net = main.Net()
test_net.load_state_dict(torch.load("saved_dicts\model-20-62.pth"))

camera = cv.VideoCapture(0)

while True:
    ret, frame = camera.read()
    frame = cv.resize(frame, (300, 300), interpolation=cv.INTER_AREA)
    if not ret:
        continue
    frame_tensor = torch.FloatTensor(frame)/255
    frame_tensor = frame_tensor[None, :]
    print(frame_tensor.shape)
    prediction = main.test_frame(test_net, frame_tensor)
    print(prediction)
    frame = cv.putText(frame, prediction, (0,250), cv.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
    cv.imshow("Cam feed", frame)
    if cv.waitKey(2) & 0xff == ord("q"):
        break

camera.release()
cv.destroyAllWindows()
