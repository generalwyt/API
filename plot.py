import numpy as np 
import matplotlib as mpl 
import argparse
#agg backend is used to create plot as a .png file
mpl.use('agg')

import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser(description='Get filename')
parser.add_argument('-f', '--file', type=str, help='Type in filename of songs with WITH extension!', required=True)
parser.add_argument('-d', '--duration', type=int, help='Type in the duration for each chunk, e.g "10" for 10 seconds', required=True)

#Read song list
def readSongs(fileName):
    with open(fileName) as f:
        content = f.readlines()
        songs =  [(line.rstrip('\n')+'.wav') for line in content]
    return songs

def calcAccuracy(a, b):
    result = []
    for i in range(len(a)):
        #print(a[i], b[i])
        result.append(np.sum(np.equal(a[i], b[i]))/len(b[i]))
    return np.sort(result)
    
def plotFigure(data, fileName, xLabel, yLabel, xAxisName):    
    #create a figure instance
    fig = plt.figure(1, figsize=(9, 6))

    #create an axes instance
    ax = fig.add_subplot(111)

    #add patch_artist=True option to ax.boxplot() to get fill color
    bp = ax.boxplot(data, patch_artist=True)

    #change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        #change outline color
        box.set( color='#7570b3', linewidth=2)
        #change fill color
        box.set( facecolor = '#1b9e77' )

    #change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)

    #change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)

    #change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)

    #change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)

    #set axis label
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_title(fileName)

    #custom x-axis labels
    ax.set_xticklabels(xAxisName)

    #remove ticks/stripe from axis
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    #save the figure
    fig.savefig(fileName, bbox_inches='tight')
    
    #clear matplotlib plotting buffer, tlo avoid drawing graphs on top of other graphs
    plt.gcf().clear()
    
if __name__ == "__main__":
    data = []
    fileName = (parser.parse_args()).file
    songs = readSongs(fileName)
    abbrSongs = ['tamborine', 'Night Owl', 'Four Seasons', 'Want', 'Really bad boy']
    print(songs)
    time = (parser.parse_args()).duration
    a = [''.join([str(time),'_',i,'_spectral_Flux_result']) for i in songs]
    b = [''.join([str(time),'_',i,'_original_message_result']) for i in songs]
    c = [''.join([str(time),'_',i,'_librosa_message_result']) for i in songs]
    
    for i in range(len(a)):
        div = 100/time
        aa = np.load(''.join([a[i], '.npy']))
        bb = np.load(''.join([b[i], '.npy']))
        arr = calcAccuracy(aa, bb)
        data.append(arr)
    
    plotFigure(data, ''.join(['Spectral Tempo Estimation Results ', str(time),'s chunks']), 'Songs', 'Accuracy', abbrSongs)
    
    data = []
    for i in range(len(a)):
        div = 100/time
        aa = np.load(''.join([a[i], '.npy']))
        bb = np.load(''.join([c[i], '.npy']))
        arr = calcAccuracy(aa, bb)
        data.append(arr)
    
    print(data)
    plotFigure(data, ''.join(['Librosa Tempo Estimation Results ', str(time),'s chunks']), 'Songs', 'Accuracy', abbrSongs)