# Optimal Distance Measure for Agglomerative Hierarchical Clustering
## CS5100 Final Project

[Github Page](https://github.com/PdAlbedo/AI_Final_Project)

Author: Xichen Liu, Jeff Turgeon\
Platform: Win11, pycharm, py3.9

- agglomerativeClustering.py: main body of processing data and compute results\
- datasetProcessing.py: pre-processing datasets\
- Results.txt: results computed\
- Analysis.xlsx: information extracted for analysis\
- Semester_Project_Proposal\
  - pdf files: past check points
- CS5100_Semester_Project_Research
  - pdf files: papers
- Datasets
  - csv files: original datasets
  - CleanedDatasets
       - csv files: cleaned datasets
  - threshold xxx: results from different clustering distance threshold xxx
  - Dataframes
       - csv files: results fomr each datasets
       - Dim: results of different number of dimensions
       - Dis: results of different diatance matrices
       - Linkage: results of different linkages

---  
#### Project processing

The inputs to the project are banch of numerical datasets.\
After we have the datasets, we cleaned them first to remove the nan values and non-necessary entries or columns.\
Then we input the datasets into the program with different linkages, distance matrices, number of dimensions in clustering.\
Number of dimensions means the different number of dimensions left in embedding space, thought the original datasets have the different number of dimensions.\
After that, the results of clustering will be output into the different data frames.
