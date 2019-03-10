To run your own experiments
1) First put the songs you want in this folder and open songs.txt, they need to be wav files with their
2) In songs.txt write down the songs you want to test for, one song for each newline
3) run runStegIbiza bt using the command: python runStegIbiza.py -f songs.txt -d 10
-f stand for the text file containing your songs
-d is the duration for each chunk
After running this command it will perform the experiment and save the resulting message, en decodings for librosa method and spectral flux method into .npy files
These files can be easily loaded using np.load('filename')
These .npy files are 2d arrays. with arrays within arrays. the inner arrays is the message decodings or encodings, the outer array are the tests to whom these decodings annd encodings belong.
4) The command will show how the experiment is progressing with the accuracies for each song.

plot.py is the file used to plot the barplots