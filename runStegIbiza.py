import librosa 
import numpy as np
import sys
import wave
import madmom
import scipy
import argparse
from progress.bar import Bar
import time as t
import matplotlib.pyplot as plt

#Set rand seed for repeatability
np.random.seed(2019)

#Set arguments for commands
parser = argparse.ArgumentParser(description='Get filename')
parser.add_argument('-f', '--file', type=str, help='Type in filename of songs with WITH extension!', required=True)
parser.add_argument('-d', '--duration', type=int, help='Type in the duration for each chunk, e.g "10" for 10 seconds', required=True)
    
#Random binary number generator
def randGen(numDraws):
    arr = [0, 1]
    rand = np.random.choice(arr, numDraws)
    #-1 to indicated first slice is going to be the reference
    return np.append(-1, rand)

#Method 1: direct bpm comparison
def dirComp(a, b):
    return 1 if a < b else 0

#Method 2: compare resultant tempo values
def resComp(a, b):
    result = []
    for i in a[:,0]:
        for j in b[:,0]:
            diff = np.float((j-i)/i)
            if abs(diff) <= 0.05:
                result.append(diff)
    
    if  np.sum(result) >= 0:
        return 1
    else:
        return 0

#Librosa tempo estimation
def libEst(stretchedChunk, sr):
    onsetEnv = librosa.onset.onset_strength(stretchedChunk, sr=sr)
    '''
    temp = librosa.beat.tempo(onset_envelope=onsetEnv, sr=sr)
    # Convert to scalar
    tempo = np.asscalar(temp)
    # Compute 2-second windowed autocorrelation
    hop_length = 512
    ac = librosa.autocorrelate(onsetEnv, 2 * sr // hop_length)
    freqs = librosa.tempo_frequencies(len(ac), sr=sr, hop_length=hop_length)
    # Plot on a BPM axis.  We skip the first (0-lag) bin.
    plt.figure(figsize=(8,4))
    plt.semilogx(freqs[1:], librosa.util.normalize(ac)[1:], label='Onset autocorrelation', basex=2)
    plt.axvline(tempo, 0, 1, color='r', alpha=0.75, linestyle='--', label='Tempo: {:.2f} BPM'.format(tempo))
    plt.xlabel('Tempo (BPM)')
    plt.ylabel('Autocorrelation strength')
    plt.grid()
    plt.title('Librosa tempo estimation')
    plt.legend(frameon=True)
    plt.axis('tight')
    plt.show()
    '''
    return temp

#Spectralflux tempo estimation
def specComp(stretchedChunk, frames, time, alpha, tMin, tMax):
    #Chop timestretched signal of 10 seconds into frames
    framedSignal = madmom.audio.signal.FramedSignal(stretchedChunk, num_frames=frames)

    #stft = madmom.audio.stft.ShortTimeFourierTransform(framedSignal, window=np.hamming(framedSignal.frame_size), circular_shift=True)
    stft = madmom.audio.stft.ShortTimeFourierTransform(framedSignal, window=np.hamming(framedSignal.frame_size))

    #make spectogram of stft for flux
    spec = madmom.audio.spectrogram.Spectrogram(stft)
    specFlux = madmom.features.onsets.spectral_flux(spec)
    '''
    plt.imshow(spec.T, aspect='auto', origin='lower')
    plt.title('Spectrogram')
    plt.xlabel('Time(Frames)')
    plt.ylabel('Frequency(Hz)')
    plt.show()
    
    plt.plot(range(len(specFlux)), specFlux)
    plt.title('Spectral flux feature vector')
    plt.xlabel('Time(Frames)')
    plt.ylabel('Amplitude')
    plt.show()
    '''
    #resonating combfilters, should work? beat activation function is not from rnn, but from other features this time.
    c = madmom.features.tempo.interval_histogram_comb(specFlux, alpha=alpha, min_tau=tMin, max_tau=tMax)
    
    #detect tempo from features
    return madmom.features.tempo.detect_tempo(c, frames/time)

#Read song list
def readSongs(fileName):
    with open(fileName) as f:
        content = f.readlines()
        songs =  [(line.rstrip('\n')+'.wav') for line in content]
    return songs
    
def stretchChunk(chunk, x):
    if x == 1:
        return librosa.effects.time_stretch(chunk, 1.01)
    elif x == 0:
        return librosa.effects.time_stretch(chunk, 0.99)
    elif x == -1:
        return chunk

def calcAccuracy(a, b):
    result = []
    for i in range(len(a)):
        #print(a[i], b[i])
        result.append(np.sum(np.equal(a[i], b[i]))/len(b[i]))
    return [np.mean(result), np.std(result)]
    
    
if __name__ == "__main__":
    fileName = (parser.parse_args()).file
    songs = readSongs(fileName)
    time = (parser.parse_args()).duration
    
    #Use 100 as base, so if duration/phi is 10 then 10 chunks, if 20 then 5 chunks etc..
    numberChunks = int(100/time)

    #number of tests again based of numbe of chunks, 10 chunks is 10 tests == 100/10, 5 chunks = 20 test 100/5, have robust results
    test = int(100/numberChunks)
    
    #progressbar
    bar = Bar("Running tests", max=(len(songs)*test))
    start = t.time()
    
    for song in songs:
        print(song)
        tests = []
        ftests = []
        ltests = []
        wf = wave.open(song, 'rb')
        #sample rate per second
        sr = wf.getframerate()

        #Open the file for reading, librosa uses int16 so 32788 divison for conversion from float to non float, in this not needed,
        #you need to specify bits in meta data
        sig, sr = librosa.load(song, offset=10, mono=True, sr=sr) 
        count = 0
        frames = 500
        
        alpha = 0.79
        tMin = int(np.floor(60*(frames/time)/240))
        tMax = int(np.floor(60*(frames/time)/40))
        
        #slices of sample rate for the chunks
        srSlices = [i for i in range(0, (sr*time*numberChunks) + (sr*time), (sr*time)-1)]
        
        for testNum in range(test):
            message = randGen(numberChunks)
            
            #init ref
            fref=[]
            lref=[]
            fdecodedMessage=[]
            ldecodedMessage=[]
            
            #loop over slices
            for i in range(len(message)):
                chunk = sig[srSlices[i]:srSlices[i+1]]
                stretchedChunk = stretchChunk(chunk, message[i])
                
                #detect librosa tempo
                ltempo = libEst(stretchedChunk, sr)
                
                #detect spectral flux tempo 
                ftempo = specComp(stretchedChunk, frames, time, alpha, tMin, tMax)

                if message[i] == -1:
                    fref = ftempo
                    lref = ltempo
                else:
                    fdecodedMessage.append(resComp(fref, ftempo))
                    ldecodedMessage.append(dirComp(lref, ltempo))
            
            #print('orginal message: ', message)
            #print('spectral flux:   ', np.append(-1, fdecodedMessage))
            #print('librosa:         ', np.append(-1, ldecodedMessage))
            #print('\n')
        
            tests.append(message[1:])
            ftests.append(fdecodedMessage)
            ltests.append(ldecodedMessage)
            bar.next()
        
        np.save(str(time)+'_'+str(song)+'_original_message_result', tests)
        np.save(str(time)+'_'+str(song)+'_spectral_Flux_result', ftests)
        np.save(str(time)+'_'+str(song)+'_librosa_message_result', ltests)
        print('\nSaved decoded messages and original message in np.arrays')
        print('spectral acc: ', calcAccuracy(tests, ftests))
        print('librosa acc:', calcAccuracy(tests, ltests))
    
    bar.finish()
    print('Done running StegIbiza tests in %s seconds' % (t.time() - start))
        