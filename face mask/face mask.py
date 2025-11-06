from ultralytics import YOLO
import cv2
from pathlib import Path

CLASSES = ['with_mask', 'without_mask', 'mask_weared_incorrect']

def detect(frame):
    results = model(frame)
    frame_after = frame.copy()
    for i,result in enumerate (results):
        #print(i,result)
        xyxy = result.boxes.xyxy
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
        for name, [x1,y1,x2,y2] in zip(names,xyxy):
            i = CLASSES.index(name) 
            color = (255,0,0) if i == 0 else (0,255,0) if i == 1 else (0,0,255)
            cv2.rectangle(frame_after, (int(x1),int(y1)), (int(x2),int(y2)), color, 3) 
    return frame_after


model_dir = Path.cwd() / 'models'

model = YOLO(model_dir / "face mask.pt")
img = cv2.imread(str(Path(r"B:\datasets\face mask\images\maksssksksss623.png")))
img = detect(img)

#cv2.imwrite('123.jpg',img)
cv2.imshow('frame', img)
cv2.waitKey(0)
cv2.destroyAllWindows()