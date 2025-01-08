from build import esim
import cv2
import numpy as np

if __name__ == '__main__':
    img = np.ones([480, 640])
    emulator = esim.EventSimulator()
    elist = emulator.generate_events(img, 0)