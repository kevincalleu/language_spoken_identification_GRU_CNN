import os
from numpy.lib import stride_tricks
import PIL.Image as Image
import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav


# Transformada de Fourier de la se;al de audio

def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)   
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    samples = np.append(samples, np.zeros(frameSize))
    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win
    
    return np.fft.rfft(frames)    
    
# Se crea una escala logaritmica para el espectro
   
def logscale_spec(spec, sr=44100, factor=20., alpha=1.0, f0=0.9, fmax=1):
    spec = spec[:, 0:256]
    timebins, freqbins = np.shape(spec)
    scale = np.linspace(0, 1, freqbins)
    scale *= (freqbins-1)/max(scale)
    newspec = np.complex128(np.zeros([timebins, freqbins]))
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = [0.0 for i in range(freqbins)]
    totw = [0.0 for i in range(freqbins)]
    
    for i in range(0, freqbins):
        if (i < 1 or i + 1 >= freqbins):
            newspec[:, i] += spec[:, i]
            freqs[i] += allfreqs[i]
            totw[i] += 1.0
            continue
        else:
            w_up = scale[i] - np.floor(scale[i])
            w_down = 1 - w_up
            j = int(np.floor(scale[i]))
           
            newspec[:, j] += w_down * spec[:, i]
            freqs[j] += w_down * allfreqs[i]
            totw[j] += w_down
            
            newspec[:, j + 1] += w_up * spec[:, i]
            freqs[j + 1] += w_up * allfreqs[i]
            totw[j + 1] += w_up
    
    for i in range(len(freqs)):
        if (totw[i] > 1e-6):
            freqs[i] /= totw[i]
    
    return newspec, freqs

# Se realiza el grafico del espectro

def plotstft(audiopath, binsize=2**10, plotpath=None, colormap="gray", alpha=1, offset=0):
    samplerate, samples = wav.read(audiopath)
    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate, alpha=alpha)
    sshow = sshow[2:, :]
    ims = 20.*np.log10(np.abs(sshow)/10e-6)
    timebins, freqbins = np.shape(ims)
    
    ims = np.transpose(ims)
    ims = ims[0:256, offset:offset+768]
    image = Image.fromarray(ims) 
    image = image.convert('L')
    image.save(plotpath)


file = open('trainingData.csv', 'r')

for iter, line in enumerate(file.readlines()[1:]):
    filepath = line.split(',')[0]
    filename = filepath[:-4]
    wavfile = 'trans.wav'
    os.system('mpg123 -w ' + wavfile + '  /home/kevin/Richi/Spoken-language-identification-master/create_png_files_from_mp3/test/' + filepath)
    alpha = np.random.uniform(0.9, 1.1)
    offset = np.random.randint(90)
    plotstft(wavfile, plotpath='/home/kevin/Richi/Spoken-language-identification-master/train_png_images/'+filename+'.'+'.png', offset=offset, alpha=1.0)
    os.remove(wavfile)
    print ("processed %d files" % (iter + 1)) 

