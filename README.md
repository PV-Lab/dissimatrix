# Dissimilarity Matrix for Perovskite Capping-Absorber Pairs
This documentation is prepared as the workflow to accompany the following study:

**"Tailoring capping-layer composition for improved stability of mixed-halide perovskites"**

Noor Titan Putri Hartono (1), Marie-Hélène Tremblay (2), Sarah Wieghold (3), Benjia Dou (1), Janak Thapa (1), Armi Tiihonen (1), Vladimir Bulovic (1), Lea Nienhaus (4), Seth R. Marder (2,5-8), Tonio Buonassisi (1), Shijing Sun (1)

Affiliations:

1. Massachusetts Institute of Technology, 77 Massachusetts Avenue, Cambridge, MA 02139
2. Georgia Institute of Technology, North Avenue, Atlanta, GA 30332
3. Argonne National Laboratory, 9700 S. Cass Avenue, Lemont, IL 60439
4. Florida State University, Department of Chemistry and Biochemistry, 95 Chieftan Way Tallahassee, FL 32306
5. University of Colorado Boulder, Renewable and Sustainable Energy Institute, Boulder, CO 80303
6. University of Colorado Boulder, Department of Chemical and Biological Engineering, Boulder, CO 80303
7. University of Colorado Boulder, Department of Chemistry, Boulder, CO 80303
8. National Renewable Energy Laboratory, Chemistry and Nanoscience Center, Golden CO 80401

## Installation and Requirements
To run the dissimilarity_matrix_all.py, Python needs to be installed (quickest way: install Miniconda https://docs.conda.io/en/latest/miniconda.html, and install Spyder by typing `
conda install -c anaconda spyder
` on Anaconda Prompt). The following packages also need to be installed.

1.  Pandas (https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html)
2.  NumPy (https://docs.scipy.org/doc/numpy/user/install.html)
3.  Seaborn (https://seaborn.pydata.org/installing.html)
4.  Matplotlib (https://matplotlib.org/users/installing.html)
5.  SciPy (https://www.scipy.org/install.html)
6.  Scikit-learn (https://scikit-learn.org/stable/install.html)

OR clone the following repository: `pip install -r requirements.txt`

## Workflow
The raw image data (.bmp files) needs to be processed first, to extract the average red, green, and blue (RGB) values over time for each samples. The dataset result is shown in DegradationData/(DegradationRun)/RGB/Calibrated folder.

Based on this data, we can calculate the dissimilarity matrix.

In dissimilarity_matrix_all.py, there are different inputs that you can change.
1. `datapoint`: How many time points you would like to include in the analysis.
2. `frequency`: How often the degradation images are taken.
3. `MAPbBrContent_1, MAPbBrContent_2, MAPbBrContent_3`: The amount of MAPbBr<sub>3</sub> of interest.
4. `concentration_1, concentration_2, concentration_3`: The concentration of capping layers of interest.
5. `annealing_1, annealing_2, annealing_3`: The annealing temperature for the capping layers of interest.
6. `capping_2, capping_3`: Capping layer materials of interest.
7. `metric`: The dissimilarity matrix distance measure, can be 'euclidean', 'cosine', or 'manhattan'.
8. `folderToSave`: Where to save the analysis results.

Run the dissimilarity_matrix_all.py file.

## Authors
| |  | 
|---|---|
|**Author(s)** | Noor Titan Putri Hartono |
|**Version** | 1.0/ August 2021  |   
|**E-mail(s)**   | noortitan at alum dot mit dot edu  |
| | |

## Attribution
This work is under an Apache 2.0 License. Please, acknowledge use of this work with the appropriate citation to the repository and research article.

## Citation

    @Misc{dissismatrix2021,
      author =   {The Dissimilarity Matrix authors},
      title =    {Dissimilarity Matrix for Perovskite Capping-Absorber Pairs},
      howpublished = {\url{https://github.com/PV-Lab/dissimatrix}},
      year = {2021}
    }