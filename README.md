# Simulator-Insight
An RFI simulation pipeline to help teach interferometry and machine learning.

Two different python files are used: 
1. Simulator.py - Consists of the class of the individual python files; 

* coordinate_conversion.ipynb: Converts coordinates to radians <br />
* uv-track.ipynb: Creates uv-tracks <br />
* source_creation.ipynb: Creates sources <br /> 
* visibility.ipynb: Creates visibilities <br />
* noise_visibility.ipynb: Adds noise to visibilities <br />
* RFI_visibilities.ipynb: Adds RFI to visibilities <br />
* classification_parameters.ipynb: Parameters to perform classification <br />
* classification_plottings.ipynb: Generates classification plots


        python Simulator.py -n 100 -a 2 -f 10 -s 5 -c 5 -r -17 -e 2 --type_classifier Naive

-n 100: Represents the number of sources that should be created. In this work, 100 sources
        was created. <br />
-a 2: Represents the pareto number (2). <br />
-f 10: Represents the field of view (10 in this case). <br />
-s 10: Represents the signal-to-noise (SNR) value of thermal noise that was added. <br />
-r -20: Represents the SNR value of RFI noise. <br />
-c 5: Represents the number of channels to be corrupted <br />
-e 2: Number of experiment to run <br />
--type_classifier Naive: Represents the type of classifier needed to be run 
                         eg. Naive, Logistic, KMeans, GMM (Exact names written here must be used) <br />


<br />
2. Restart.py - Used to repeat the experiment based on the number of times it has to be done

        python Restart.py -n 100 -a 2 -f 10 -s 5 -c 5 -r -17 -e 2 --type_classifier Naive 
* Uses the same parameters defined above but best used when running several experiments.
