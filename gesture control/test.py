import numpy as np
import cv2 as cv
from ultralytics import YOLO

def gesture(recording = False):
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print('Камера недоступна')
        return
    

    if recording:
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        fps = 20.0
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        out = cv.VideoWriter('output.avi', fourcc, fps, (width, height))

    ret, frame = cap.read()

    model = YOLO("yolo11n.pt")

    while cap.isOpened():
        ret, frame = cap.read()
        #frame = cv.resize(frame, (640, 480))

        if not ret:
            break
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        
        
        results = model(frame)
        frame_after = frame.copy()
        for i,result in enumerate (results):
            xyxy = result.boxes.xyxy
            for ix,iy,x,y in xyxy: 
                ix = int(ix)
                iy = int(iy)
                x = int(x)
                y = int(y)
                cv.rectangle(frame_after,(ix,iy),(x,y),(0,255,0),5)

        cv.imshow('frame', frame)
        cv.imshow('frame_after', frame_after)
        
        if recording:
            out.write(frame)
    cap.release()
    if recording:
            out.release()
            
    cv.destroyAllWindows()


if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    #gesture(recording=True)
    '''
    test_rectangles=[[1,2,5,6],
                     [4,3,5,8],
                     [20,20,2,2]]
    print(f'Итоговые прямоугольники {unite_rectangles(test_rectangles)}')
    '''
