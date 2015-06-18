import cv2
import math
import numpy as np
import pyaudio
import itertools
from scipy import interpolate
from operator import itemgetter
from matplotlib import pyplot as plt

# Audio parts were borrowed from
# http://davywybiral.blogspot.be/2010/09/procedural-music-with-pyaudio-and-numpy.html
# Video parts are based on various opencv tutorials

def sine(frequency, length, rate):
    length = int(length * rate)
    factor = float(frequency) * (math.pi * 2) / rate
    return np.sin(np.arange(length) * factor)

def harmonics1(freq, length):
    a = sine(freq * 1.00, length, 44100)
    b = sine(freq * 2.00, length, 44100) * 0.5
    c = sine(freq * 4.00, length, 44100) * 0.125
    return (a + b + c) * 0.2

def audiospectrum(hist):
    duration = 0.01
    rate = 44100
    frequency = 440
    x = np.linspace(0,duration*rate,duration*rate)
    y = np.linspace(220, 440, len(hist))
    xv,yv = np.meshgrid(x,y)

    # these are the waves we need to compose
    sines = np.sin(xv*yv*math.pi*2/rate)
    
    amplitude = np.repeat(np.array(hist)[:,np.newaxis], duration*rate, axis=1)
    # set the amplitudes according to the histogram values
    return sum(amplitude*sines)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=44100, output=1)

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #print sum of value (indicating brightness)
    #print sum(hsv[:,:,2].ravel())

    # Add polylines containing histogram
    # 0 = hue, 1 = saturation, 2 = value
    hist = cv2.calcHist( [hsv], [0], None, [256], [0, 256] )

    vals = hist[:,0]
    histpts = np.zeros((len(hist),2))
    histpts[:,0] = np.arange(len(hist))
    histpts[:,1] = vals / frame.shape[0]

    # Play audio sample
    samples = audiospectrum(vals)
    stream.write(samples.astype(np.float32).tostring())

    cv2.polylines(frame, np.int32([histpts]), False, (255,255,255))

    # Display the resulting frame
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture, close the audiostream
cap.release()
cv2.destroyAllWindows()

stream.close()
p.terminate()
