import numpy as np
from datetime import datetime  
import cv2 as cv
from matplotlib import pyplot as plt

def unite_rectangles(rectangles):
    
    # тут лежат объединенные прямоугольники 
    unite = []
    for  (x, y, w, h) in rectangles: 
        change = False
        for i, (_x, _y, _w, _h) in enumerate(unite):
            # используем координаты уже имеющегося прямоугольника как начало координат
            x_m = x - _x
            y_m = y - _y
            
            # если левая верхняя точка или правая нижняя нового прямоугольника внутри имеющегося
            if (0 < x_m < _w and 0 < y_m < _h) or (0 < x_m + w < _w and 0 < y_m + h < _h):         
                
                # берем максимально возможный прямоугольник из двух
                change = True
                new_x = _x if _x < x else x
                new_y = _y if _y < y else y
                if _x + _w > x + w:
                    new_w = _x + _w - new_x
                else:
                    new_w = x + w - new_x
                
                if _y + _h > y + h:
                    new_h = _y + _h - new_y
                else:
                    new_h = y + h - new_y
                unite[i] = new_x, new_y, new_w, new_h
                break
            
        # если не было изменений, то добавляем в список
        if not change:
            unite.append([x, y, w, h])
    return unite

def detect_movement():
    cap = cv.VideoCapture(0)
  
    if not cap.isOpened():
        print('Камера недоступна')
        return
    
    ret, frame = cap.read()
    gray2 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray2 = cv.GaussianBlur(gray2, (31, 31), 0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        
            
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (5, 5), 0)

        diff = cv.absdiff(gray, gray2)

        diff = cv.dilate(diff, None, iterations=2)

        kernel = np.ones((9,9),np.uint8)
        diff = cv.morphologyEx(diff, cv.MORPH_OPEN, kernel)
        diff = cv.morphologyEx(diff, cv.MORPH_CLOSE, kernel)
        ret,thresh = cv.threshold(diff,25,255,cv.THRESH_BINARY)
        #thresh = cv.adaptiveThreshold(diff,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
        #    cv.THRESH_BINARY,11,2)


        contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        rectangles = []
        for i, cnt in enumerate(contours):
            if cv.contourArea(cnt) < 500:  continue
            (x, y, w, h) = cv.boundingRect(cnt)
            rectangles.append([x, y, w, h])

        #Объединение прямоугольников в группы
        rectangles = unite_rectangles(rectangles)
        for  (x, y, w, h) in rectangles:   
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(frame, "Movement", (x, y-10), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
        gray2 = gray
        cv.imshow('frame', frame)
        cv.imshow('Threshold', thresh)
        cv.imshow('Diff', diff)
        #out.write(frame)
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    detect_movement()
    '''
    test_rectangles=[[1,2,5,6],
                     [4,3,5,8],
                     [20,20,2,2]]
    print(f'Итоговые прямоугольники {unite_rectangles(test_rectangles)}')
    '''
