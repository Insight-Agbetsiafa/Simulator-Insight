import Simulator		#Enter name of file (eg. Simulator)
from Simulator import main   
from Simulator import sim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
import re
import random
import sys, getopt
import os

import matplotlib
import itertools 
import matplotlib.cm as cm
import pickle
import sklearn

from sklearn.linear_model import LogisticRegression as logis
from sklearn.metrics import confusion_matrix
from sklearn.neighbors._nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap


num_sources, SNR,SNR_high,fov,pareto_number,num_channels_to_corrupt,number_of_experiment,type_classifier=main()
def rerun():   
    counter = 1
    while counter <= number_of_experiment:
        exec(open("Simulator.py").read()) 	#Enter script name
        counter = counter+1
    else:
        print("Done with " +str(number_of_experiment) +" experiments")
        sys.exit()
    

if __name__ == '__main__':
     rerun()
    

#python Restart.py -n 100 -s 30 -r -20 -a 2 -f 10 -c 5 -e 2 --type_classifier Naive   

