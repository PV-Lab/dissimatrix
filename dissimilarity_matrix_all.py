# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 10:50:59 2021

@author: Noor Titan Hartono
internal version: 9

Steps in computing dissimilarity matrix (for EACH DATA POINT):
    1. Load the data, sort them, drop the NA rows.
    2. Select the data of interest (certain absorbers, certain capping layers
                                    with certain annealing T and concentrations)
    3. Calculate the dissimilarity matrix using scikit-learn: pairwise distances.
    4. Generate figures, and save them in specific path.

"""

# IMPORTING LIBRARIES
# import scipy.io as sio
# import numpy.matlib as nm
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib as mpl
import os
import pandas as pd
from collections import OrderedDict
cmaps = OrderedDict()
import seaborn as sns

#%% User inputs

"""
INPUT FOR THE CODE

datapoint: How many data points you want to include in the analysis, default: 341.

frequency: How often the degradation images are taken (every... minutes).

metric: Dissimilarity matrix metric, possible types: 'cosine', 'euclidean',
                or 'manhattan'.
                
MAPbBrContent: Br amount in the absorber (between 0.0-1.0). In this study,
                the options are: 0, 0.25, 0.5, and 1.0.
                
concentration:  Concentration of capping layer precursor.

annealing: Annealing temperature of capping layer film.

capping: Capping layer meaterial being analyzed. For 

folderToSave: Where to save the dissimilarity matrix results, within the 'Results'
                folder. 
                

"""

datapoint = 341 #341 # How many data points you want to include in the analysis
frequency = 3 # How often the degradation images are taken (every .... minutes)

# If we want to compare 3 different data: bare film, PTEAI-based film (10mM, 100C annealing),
# and 9-Cl-capped film of I : Br = 3 : 1, we can input the following:

# Bare films
MAPbBrContent_1 = 0
concentration_1 = 0 # Because it's bare film, both concentration and annealing (for capping) is 0s.
annealing_1 = 0

# PTEAI-capped films
MAPbBrContent_2 = 0
concentration_2 = 10 # 10 mM
annealing_2 = 100 # 100C
capping_2 = "PTEAI"

# 9-Cl-capped films
MAPbBrContent_3 = 0
concentration_3 = 10 # 10 mM
annealing_3 = 100 # 100C
capping_3 = "09"

# Dissimilarity matrix metric: euclidean, manhattan, cosine
metric = "cosine" 

folderToSave = "MAPIBr_1_0"

#%% Functions

currentDir = os.path.dirname(os.path.realpath(__file__))

def combineRGB(location, datapoint, frequency):
    """
    Combine sliced RGB data according to desired time, with the
    sample name, capping type, concentration, annealing T.
    - location: desired degradation folder name
    - datapoint: how many datapoints you want to include
    - frequency: the degradation image is taken every ... minutes
    - RETURN: the giant RGB data, with NaN dropped
    """
    R = pd.read_csv(os.path.join(location,'sample_r_cal.csv'),header=None)
    G = pd.read_csv(os.path.join(location,'sample_g_cal.csv'),header=None)
    B = pd.read_csv(os.path.join(location,'sample_b_cal.csv'),header=None)
    samplename = pd.read_csv(os.path.join(location,'Samples_cap.csv'))
    
    # Subtracting data with the initial RGB value of mean(first 11 data points)
    subtract_datapoint = 11
    subtract_R = np.expand_dims(np.mean(R.iloc[:,:subtract_datapoint], axis=1),axis=1)
    subtract_G = np.expand_dims(np.mean(G.iloc[:,:subtract_datapoint], axis=1),axis=1)
    subtract_B = np.expand_dims(np.mean(B.iloc[:,:subtract_datapoint], axis=1),axis=1)
    subtract_all = np.expand_dims(np.amin(np.concatenate((subtract_R, subtract_G,
                                        subtract_B), axis=1), axis=1),axis=1)
    
    # Combining RGB 
    RGB = pd.concat([R.iloc[:,:datapoint],
                     G.iloc[:,:datapoint],
                     B.iloc[:,:datapoint]],axis=1)
    
    RGB_sub = RGB.subtract(subtract_all, axis=1)
    
    # Combine sample name with RGB, drop rows with NaN values
    RGB_samplename = (pd.concat([samplename, RGB_sub], axis=1)).dropna(axis=0)
    
    # Sort by Capping column, then concentration, if preferred
    # RGB_samplename_sorted = RGB_samplename.sort_values(by=['Capping', 'Concentration', 'Annealing'])
    
    return (RGB_samplename)

#%% Setting up figures and colormaps

if metric=="cosine":
    cmap = sns.cubehelix_palette(start=.5, rot=-.75, reverse=True, as_cmap=True) # colormap for cosine
    colorMax = 0.2
    colorMin = 0
elif metric=="euclidean":
    cmap = sns.cubehelix_palette(as_cmap=True, reverse=True)
    colorMax = 1000
    colorMin = 0
else:
    cmap = sns.cubehelix_palette(start=2, rot=.25, reverse=True, as_cmap=True)
    colorMax = 30000
    colorMin = 0

#%% Generating dissimilarity matrix

# Looping for the time range of interest, calculating for dissimilarity matrix
# at each time range

for i in range(datapoint):
    
    # LOADING RGB DATA
    os.chdir(currentDir+'/Dataset/')
    
    # LOADING RGB DATA, 100% MAPI using July samples
    MAPIBr_1_0_1to5 = combineRGB('20200714-R1-TH',i+1,frequency)
    MAPIBr_1_0_6to10 = combineRGB('20200715-R1-TH',i+1,frequency)
    MAPIBr_1_0_2nd = combineRGB('20201001-R1-JT',i+1,frequency)
    
    MAPIBr_3_1_1to5 = combineRGB('20200710-R1-JT',i+1,frequency)
    MAPIBr_3_1_6to10 = combineRGB('20200720-R1-TH',i+1,frequency)
    MAPIBr_3_1_2nd = combineRGB('20201005-R1-TH',i+1,frequency)
    
    MAPIBr_1_1_1to5 = combineRGB('20200707-R1-JT4TH',i+1,frequency)
    MAPIBr_1_1_6to10 = combineRGB('20200721-R1-JT',i+1,frequency)
    MAPIBr_1_1_2nd = combineRGB('20201006-R1-TH',i+1,frequency)
    
    MAPIBr_1_3_1to5 = combineRGB('20200724-R1-JT',i+1,frequency)
    MAPIBr_1_3_6to10 = combineRGB('20200812-R1-TH',i+1,frequency)
    MAPIBr_1_3_2nd = combineRGB('20201008-R1-TH',i+1,frequency)
    
    MAPIBr_0_1_1to5 = combineRGB('20200730-R1-TH',i+1,frequency)
    MAPIBr_0_1_6to10 = combineRGB('20200814-R1-TH',i+1,frequency)
    MAPIBr_0_1_2nd = combineRGB('20201013-R1-JT',i+1,frequency)
    
    # CONCATENATING all

    deg_all_samplename = pd.concat([MAPIBr_1_0_1to5, MAPIBr_1_0_6to10, MAPIBr_1_0_2nd,
                                    MAPIBr_3_1_1to5, MAPIBr_3_1_6to10, MAPIBr_3_1_2nd,
                                    MAPIBr_1_1_1to5, MAPIBr_1_1_6to10, MAPIBr_1_3_1to5, 
                                    MAPIBr_1_1_2nd, MAPIBr_1_3_6to10, MAPIBr_1_3_2nd,
                                    MAPIBr_0_1_1to5, MAPIBr_0_1_6to10, MAPIBr_0_1_2nd],
                                    axis=0)
    
    # SORTING based on absorbers first OR capping first
    
    # Sorting based on absorbers first
    sorted_absorber = deg_all_samplename.sort_values(by=['MAPbBr', 'Capping', 
                                                     'Concentration', 'Annealing'])
    
    # Sorting based on capping layer first
    sorted_capping = deg_all_samplename.sort_values(by=['Capping', 'MAPbBr', 
                                                     'Concentration', 'Annealing'])
    
    # DROPPING based on certain values

    sorted_absorber_1 = sorted_absorber[((sorted_absorber['MAPbBr'] == MAPbBrContent_1) &
                                          (sorted_absorber['Concentration'] == concentration_1) &
                                          (sorted_absorber['Annealing'] == annealing_1))]
    
    sorted_absorber_2 = sorted_absorber[((sorted_absorber['MAPbBr'] == MAPbBrContent_2) &
                                        (sorted_absorber['Concentration'] == concentration_2) &
                                        (sorted_absorber['Annealing'] == annealing_2) &
                                        (sorted_absorber['Capping'] == capping_2))]
    
    sorted_absorber_3 = sorted_absorber[((sorted_absorber['MAPbBr'] == MAPbBrContent_3) &
                                        (sorted_absorber['Concentration'] == concentration_3) &
                                        (sorted_absorber['Annealing'] == annealing_3) &
                                        (sorted_absorber['Capping'] == capping_3))]
    
    # COMBINING all the sorted_absorber data
                                            
    sorted_absorber_10mM_100C = pd.concat([sorted_absorber_1, sorted_absorber_2,
                                          sorted_absorber_3], axis=0)
                                            
    
    # CALCULATING the dissimilarity matrix
    from sklearn.metrics.pairwise import pairwise_distances

    d_absorber = pairwise_distances(sorted_absorber_10mM_100C.drop(sorted_absorber_10mM_100C.iloc[:, 0:9], 
                                           axis = 1), metric=metric) # The first 9 being dropped because they're not actual data

    # Masking upper triangle
    mask_ut = np.triu(np.ones_like(d_absorber,dtype=bool))    
    
    # VISUALIZING dissimilarity matrix
    
    # Defining where to save the figure
    os.chdir(currentDir+'/Results/'+metric+'/'+folderToSave+'/bare_cap9_PTEAI/')
   
    # Plot the dissimilarity matrix
    fig = plt.figure(i+1,figsize=(8,6),dpi=300) # for all use 10,10; 8,6 for subsets
    ax = fig.add_subplot(111)
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 24 # for all: 20, 24 for subsets
    ax = sns.heatmap(d_absorber, mask=mask_ut, cmap=cmap, vmax=colorMax,
                vmin=colorMin, square=True)
    
    ax.set_xlabel('Samples') # Labeling x-axis
    ax.set_ylabel('Samples') # Labeling y-axis
    
    # Labeling x-ticks and y-ticks
    if MAPbBrContent_1 == 0:
        # For just I (0 Br) films
        plt.xticks(np.array([5.5,13,16]),('Bare','PTEAI','9-Cl'))
        plt.yticks(np.array([6,13,16.5]),('Bare','PTEAI','9-Cl'))
        # plt.xticks(np.array([5.5,13]),('Bare','PTEAI'))
        # plt.yticks(np.array([6,13]),('Bare','PTEAI'))
    else:
        # For any Br-mixed films 
        plt.xticks(np.array([6,14,17]),('Bare','PTEAI','9-Cl'))
        plt.yticks(np.array([7,14,17.5]),('Bare','PTEAI','9-Cl'))
    
    # Saving the figure
    fig.savefig('barecap9PTEAI_'+str(f'{i:03}')+'.png') 
    
    # Closing the figure
    plt.close('all')

