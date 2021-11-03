# Simulator-Insight
An RFI simulator to help teach interferometry and machine learning.

Two different python files are used: 
1. Simulator.py - Consists of the class of the individual python files; 

coordinate_conversion.ipynb: Converts coordinates to radians
print ("\n")
uv-track.ipynb: Creates uv-tracks
print ("\n")
source_creation.ipynb: Creates sources
visibility.ipynb: Creates visibilities
noise_visibility.ipynb: Adds noise to visibilities
RFI_visibilities.ipynb: Adds RFI to visibilities 
classification_parameters.ipynb: Parameters to perform classification
classification_plottings.ipynb: Generates classification plots


2. Restart.py - Used to repeat the experiment based on the number of times it has to be done.


python Simulator.py -n 100 -a 2 -f 10 -s 10 -c 5 -r -20 -e 2 --type_classifier Naive

-n 100: Represents the number of sources that should be created. In this work, 100 sources
        was created.
-a 2: Represents the pareto number (2).
-f 10: Represents the field of view (10 in this case).
-s 10: Represents the signal-to-noise (SNR) value of thermal noise that was added.
-r -20: Represents the SNR value of RFI noise.
-c 5: Represents the number of channels to be corrupted
-e 2: Number of experiment to run 
--type_classifier Naive: Represents the type of classifier needed to be run 
                         eg. Naive, Logistic, KMeans, GMM


python Restart.py -n 100 -a 2 -f 10 -s 10 -c 5 -r -20 -e 2 --type_classifier Naive
Uses the same parameters defined above but best used when running several experiments.
