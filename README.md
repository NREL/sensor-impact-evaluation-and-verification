[![CircleCI](https://circleci.com/gh/circleci/circleci-docs.svg?style=svg)](https://circleci.com/gh/circleci/circleci-docs)
[![CircleCI](https://circleci.com/gh/circleci/circleci-docs.svg?style=shield)](https://circleci.com/gh/circleci/circleci-docs)

# Python Package: Sensor Impact FDD (SIF)
This is an NREL public repository used for sensor impact evaluation and verification project funded by DOE. Specifically, the function of repository can evaluate the sensor impact, including sensor inaccuracy, sensor selection, and sensor cost analysis on fault detection and diagnostics (FDD) performance.

Authors: Liang Zhang, Matt Leach, National Renewable Energy Laboratory, January 18, 2021

## Installation
Download and install the latest version of [Conda](https://docs.conda.io/en/latest/) (version 4.4 or above)
Create a new conda environment:

`$ conda create -n <name-of-repository> python=3.8 pip`

`$ conda activate <name-of-repository>`

(If you’re using a version of conda older than 4.4, you may need to instead use source activate <name-of-repository>.)

Make sure you are using the latest version of pip:

`$ pip install --upgrade pip`

Install the environment needed for this repository:

`$ pip install -e .[dev]`

## Technical Route: General Guidance and User Defined Analysis
The run of "General Guidance" mode in this repo relies on the fault building simulation data that calibrated on Oak Ridge National Laboratory's Flexible Research Platform. Check the Data section for download information.

The run of "User Defined Analysis" in this repo relies on data generated by users, and the guidance to generate simulation data can be found in documents/User_Defined_Analysis_Simulation_Data_Generation_Guidance.md. This documentation is a work-in-progress and the estimated finishing time is June 2022 (FY2022 Q3).

## Data
This module is developed based on the fault building simulation data that calibrated on Oak Ridge National Laboratory's Flexible Research Platform.
The data include 22 fault types data under seven weathers from different climate zones. The details of physics-based modeling of the faulty buildings are introduced in [this paper](https://www.mdpi.com/2075-5309/9/11/233/htm).

Downloading data is mandatory to use the classes in the repo,

The data can be downloaded [here](https://docs.conda.io/en/latest/) (This is temporarily a dummy link. Finding a server to host the data). Please contact Liang.Zhang@nrel.gov if there is any download issues.

## Modules
The base.py and sensor_impact_FDD.py provide modules to realize sensor impact evaluations. This repo contains three sub-classes that realize three modules used for sensor impact evaluation in FDD.

### Module 1: Sensor Selection Analysis
Sub-Class Name: sensor_selection_analysis

Inputs: `feature_extraction_config` defines feature extraction options. `feature_selection_config` defines feature selection options such as filter, wrapper, and hybrid method, `by_fault_type` defines whether the analysis is conducted by fault type or consider all faults in one model. `top_n_features` defines number of the most important features shown as results, `rerun=False` defines whether rerun the analysis if running results are detected.

Output: most important sensors rank by importance (optionally by fault type)

### Module 2: Sensor Inaccuracy Analysis
Sub-Class Name: sensor_inaccuracy_analysis

Inputs: `sensor_fault_probability_table` defines conditional probability of sensor fault as the key input for this analysis. `by_fault_type` defines whether the analysis is conducted by fault type or consider all faults in one model. `top_n_features` defines number of the most important features shown as results.`Monte_Carlo_runs` defines number of runs for Monte Carlo simulation. `rerun=False` defines whether rerun the analysis if running results are detected.

Output: most important sensors rank by importance with uncertainty (optionally by fault type) and FDD accuracy Kernel density estimation (KDE) plot which quantifies the uncertainty of FDD accuracy.

### Module 3: Sensor Cost Analysis
Sub-Class Name: sensor_cost_analysis

Inputs: `analysis_mode` defines whether the analysis focuses on single sensor or sensor group. `baseline_sensor_set` defines the existing sensors as the starting point of the analysis. `candidate_sensor_set` defines the candidate sensor set to be evaluated. `rerun` defines whether rerun the analysis if running results are detected.

Output: summary table ranking sensor opportunity values (Sensor Threshold Marginal Cost, or STMC) by importance, with disaggregated STMC values for energy efficiency, thermal comfort, and maintenance.

## Use Cases
Create an object to define the general analysis configurations. In the example, the technical route is "general guidance", which indicates that the analysis is based on providing general guidance and analysis instead of analyzing specific building. In this mode, the users do not need to prepare data for the analysis for themselves.

`$import senosr_impact_FDD as SIF

import base_functions as bf

import os

example_object = SIF.SensorImpactFDD(technical_route='general_guidance',
                                     building_type_or_name='small_commercial_building',
                                     ml_algorithm='random_forest',
                                     weather='TN_Knoxville',
                                     root_path=os.getcwd())`

After creating the object, three kinds of anaylsis can be done based on the object. The first analysis is sensor selection analysis which can be used to identify important sensors for FDD purpose. There are two major modes in this anaylsis (1) important sensor by fault type, which ranks sensors by each fault, (2) important sensors for all fault types.

`$example_object.sensor_selection_analysis(feature_extraction_config=[['mean', 'sum', 'std', 'skew', 'kurt'], '4h'],
                                         feature_selection_config=[['filter', 'pearson', 0.5], ['wrapper', 'forward'],
                                                                   ['embedded', False]],
                                         by_fault_type=False,
                                         top_n_features=3,
                                         rerun=False)`

The example result is shown as follows:

![Module 1 Results Example](https://github.com/NREL/sensor-impact-evaluation-and-verification/blob/main/figures/module_1_results_example.png)

The second analysis is sensor inaccruacy impact on sensor selection and FDD performance. The key input to this module is sensor fault probability table to define the probability of sensor with faults.

`$example_object.sensor_inaccuracy_analysis(sensor_fault_probability_table=bf.Base.sensor_fault_probability_table,
                                          by_fault_type=False,
                                          top_n_features=20,
                                          Monte_Carlo_runs=1000,
                                          rerun=False)`

The example result is shown as follows:

![Module 2 Results Example](https://github.com/NREL/sensor-impact-evaluation-and-verification/blob/main/figures/module_2_results_example.png)

The third analysis is sensor cost analysis. The key inputs to be defined in this module is baseline sensor set, candidate sensor set, and analysis mode which defines whether sensors are evaluate one by one or group by group.

`$example_object.sensor_cost_analysis(analysis_mode='group',
                                    baseline_sensor_set=bf.Base.baseline_sensor_set,
                                    candidate_sensor_set=bf.Base.candidate_sensor_set,
                                    rerun=True)`

The example result is shown as follows:  

![Module 3 Results Example](https://github.com/NREL/sensor-impact-evaluation-and-verification/blob/main/figures/module_3_results_example.png)

## Machine Learning Algorithms Options in this module
### Linear Models
[Classifier using Ridge regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html#sklearn.linear_model.RidgeClassifier)

[Logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)

[Stochastic Gradient Descent - SGD](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier)

[Passive Aggressive Algorithms](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html#sklearn.linear_model.PassiveAggressiveClassifier)

[Perceptron](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html#sklearn.linear_model.Perceptron)

### Linear and Quadratic Discriminant Analysis
[Linear Discriminant Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis)

[Quadratic Discriminant Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html#sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis)

### Support Vector Machines
[C-Support Vector Classification](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)

[Nu-Support Vector Classification](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html#sklearn.svm.NuSVC)

[Linear Support Vector Classification](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC)

### Nearest Neighbors
[Classifier implementing the k-nearest neighbors vote](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)

[Nearest centroid classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html#sklearn.neighbors.NearestCentroid)

[Gaussian process classification (GPC) based on Laplace approximation](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html#sklearn.gaussian_process.GaussianProcessClassifier)

### Naive Bayes
[Gaussian Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB)

[Naive Bayes classifier for multivariate Bernoulli models](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB)

[Naive Bayes classifier for multinomial models](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB)

### Decision Trees
[Decision Tree Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)

### Ensemble methods
[Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier)

[Extra-Trees Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier)

[AdaBoost classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier)

[Histogram-based Gradient Boosting Classification Tree](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html#sklearn.ensemble.HistGradientBoostingClassifier)

[Gradient Boosting for classification](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier)

### Neural Network
[Multi-layer Perceptron classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier)

## Work in Progress
~~TODO: find a better structure to select ML algorithm, 2/22/2022~~

~~TODO: documentation to describe how to generate simulation data when technical_route is user_defined_analysis, 2/22/2022~~

~~TODO: reduce the size of temporary files generated in sensor_inaccuracy_analysis module, 2/25/2022~~

~~TODO: improve the computation efficiency of all three modules, 2/30/2022~~

~~TODO: use config.json to define case-specific parameters instead of put all of constants in base_functions.py, 2/30/2022~~

~~TODO: improve sensor cost analysis using sensor selection module results instead of a simple feature selection, 3/4/2022~~

TODO: Find a server to store simulation data used for "general_guidance" mode

TODO: use fault prevalence probability to calculate STMC ranges

TODO: Write documentation for preparing simulation data in the "User Defined Analysis" mode

## Improvement and Test in Progress (FY22 Q3 AOP Tasks)
Test: Missing data, NAN, wrong data type, super large data in the simulation file

Solution: add data cleaning function to remove sensor with wrong data type and remove data with bad quality

Test: Monte Carlo Simulation number of runs is set too large

Solution: add a function to estimate the calculation time and raise a warning

Test: Certain Machine Learning algorithm is very inaccurate and the analysis results are very unreasonable

Solution: raise a warning based on ML validation error threshold
