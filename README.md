 Comparing classification algorithms for assignment of chemical compounds into different classes

 The goal of computer-aided drug design is identification of 
 novel compounds active against selected protein targets. In general, 
 all ligand-based virtual screening methods are based on the searching of 
 ligand similarity by comparison molecular structure descriptors and properties. 
 Here, I will show how to use chemical descriptor in categorising molecules 
 based on their biological functions. These features can classify compounds 
 that are diverse in substructure but nonetheless bind to the same 
 macromolecular binding sites, and can therefore be used to prepare molecular 
 databases for high-throughput and virtual screening.

 We will implement the most common classification algorithms in scikit-learn and
 compare their performance.

 The following algorithms will be compared:

 - Naive Bayes
 - Support Vector Machine
 - Logistic Regression
 - K-Nearest Neighbors
 - Linear Discriminant Analysis
 - Support Vector Machine
 - Decision Tree Classifier

 The dataset and chemical descriptors are described in my article [link.....], where 
 I used a linear discriminant analysis to assign chemical compounds into 7 different 
 classes.
 Here we will use a smaller dataset containing 45 molecules that are known to bind
 the cyclooxygenase 1 (COX1) enzyme, 59 molecules that bind HIV-1 protease and 41 molecules bind 
 Cytochrome C peroxidase enzyme.
 Our dataset has three classes and eight numeric input variables (chemical descriptors) of varying scales.

 Let's load an input data and print the first 5 elements. The molecules binding COX-1 enzyme have class label '1',
 the molecules binding HIV-1 protease have class label '2' and molecules binding Cytochrome C peroxidase enzyme has
 class label '3'. The eight chemical descriptors labeled as D1-D8.

import pandas as pd
data = pd.read_csv('Dataset_COX-1_HIV-1_Cyt.csv')
data[0:5]
