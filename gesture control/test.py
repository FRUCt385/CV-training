import numpy as np
import cv2 as cv
from ultralytics import YOLO
import time

import pycaw.pycaw as pycaw
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL


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

    time_t = time.time()

    frame_after = frame.copy()

    while cap.isOpened():
        ret, frame = cap.read()
        #frame = cv.resize(frame, (640, 480))

        if not ret:
            break
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        
        if time.time() - time_t >= 1:
            print(1)
            time_t = time.time()
        
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
        
        
        cv.imshow('frame_after', frame_after)
        cv.imshow('frame', frame)
        
        
        if recording:
            out.write(frame)
    cap.release()
    if recording:
            out.release()
            
    cv.destroyAllWindows()


# можно использовать для записи действий pyautogui

import pycaw.pycaw as pycaw
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

def set_system_volume(level):
    """Установить уровень громкости (0.0 - 1.0)"""
    devices = pycaw.AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        pycaw.IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(pycaw.IAudioEndpointVolume))
    volume.SetMasterVolumeLevelScalar(level, None)

def get_system_volume():
    """Получить текущий уровень громкости"""
    devices = pycaw.AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        pycaw.IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(pycaw.IAudioEndpointVolume))
    return volume.GetMasterVolumeLevelScalar()


if __name__ == "__main__":
    '''
    import subprocess
    from subprocess import call

    a = call('tg.bat')
    if a == 0: print('Успешно!')
    else: print('Не выполнено(')
    '''
    
    '''  
    while True:
        volume = get_system_volume()
        a = int(input())
        if a == 0: UP = False
        if a == 1: UP = True
        if a == 2: break
        volume = volume + 0.1 if UP else volume - 0.1
        volume = 1 if volume > 1 else volume
        volume = 0 if volume < 0 else volume

        set_system_volume(volume)

        print(f"Текущая громкость: {volume}")
    '''
    model = YOLO("yolo11n.pt")
    #gesture(recording=False)

