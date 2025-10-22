import numpy as np
import cv2 as cv

# rectangles - список с прямоугольниками
# depth - расстояние между прямоугольниками в пикселях при котором считается, что они имеют общую площадь, по умолчанию = 0
def unite_rectangles(rectangles, depth = 0):
    
    # тут лежат объединенные прямоугольники 
    unite = []
    for  (x, y, w, h) in rectangles: 
        change = False
        for i, (_x, _y, _w, _h) in enumerate(unite):
            # используем координаты уже имеющегося прямоугольника как начало координат
            x_m = x - _x
            y_m = y - _y
            
            # если левая верхняя точка или правая нижняя нового прямоугольника внутри имеющегося
            if (0 < x_m < _w + depth and 0 < y_m < _h + depth) or (0 < x_m + w < _w + depth and 0 < y_m + h < _h + depth):         
                
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

def detect_movement(depth = 0, recording = False):
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

    bg_sub = cv.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
    kernel = np.ones((3,3),np.uint8)


    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv.resize(frame, (640, 480))

        if not ret:
            break
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        
        fg_mask = bg_sub.apply(frame)
        ret,fg_mask_thresh = cv.threshold(fg_mask,180,255,cv.THRESH_BINARY)
        
        fg_mask_thresh1 = cv.morphologyEx(fg_mask_thresh, cv.MORPH_OPEN, kernel)
        fg_mask_thresh = cv.morphologyEx(fg_mask_thresh1, cv.MORPH_CLOSE, kernel)

        contours, _ = cv.findContours(fg_mask_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        rectangles = []

        for i, cnt in enumerate(contours):
            if cv.contourArea(cnt) < 500:  continue
            (x, y, w, h) = cv.boundingRect(cnt)
            rectangles.append([x, y, w, h])

        #Объединение прямоугольников в группы
        rectangles = unite_rectangles(rectangles, depth)
        for  (x, y, w, h) in rectangles:   
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(frame, "Movement", (x, y-10), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    

        cv.imshow('frame', frame)

        if recording:
            out.write(frame)
    cap.release()
    if recording:
            out.release()
            
    cv.destroyAllWindows()


if __name__ == "__main__":
    
    detect_movement(recording=True)
    '''
    test_rectangles=[[1,2,5,6],
                     [4,3,5,8],
                     [20,20,2,2]]
    print(f'Итоговые прямоугольники {unite_rectangles(test_rectangles)}')
    '''
