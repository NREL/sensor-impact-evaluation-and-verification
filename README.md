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

The run of "User Defined Analysis" in this repo relies on data generated by users, and the guidance to generate simulation data can be found in `documents/User_Defined_Analysis_Simulation_Data_Generation_Guidance.docx`.

## Data
This module is developed based on the fault building simulation data that calibrated on Oak Ridge National Laboratory's Flexible Research Platform.
The data include 22 fault types data under seven weathers from different climate zones. The details of physics-based modeling of the faulty buildings are introduced in [this paper](https://www.mdpi.com/2075-5309/9/11/233/htm).

The users need to prepare the data themselves only when the “technical_route” is “user_defined_analysis”. The guidance of data preparation locates here: `documentations/User_Defined_Analysis_Simulation_Data_Preparation_Guidance.docx`. If “technical_route” is “general_guidance”, the module will provide an automatic download of simulation data, so the users do not need to prepare simulation data. Please contact Liang.Zhang@nrel.gov if there is any download issues.

## Modules
The sensor_impact_FDD.py provide modules to realize sensor impact evaluations. This repo contains three sub-classes that realize three modules used for sensor impact evaluation in FDD.

### Module 1: Sensor Selection Analysis
Sub-Class Name: sensor_selection_analysis

Inputs: `feature_extraction_config` defines feature extraction options. `feature_selection_config` defines feature selection options such as filter, wrapper, and hybrid method, `by_fault_type` defines whether the analysis is conducted by fault type or consider all faults in one model. `top_n_features` defines number of the most important features shown as results, `rerun=False` defines whether rerun the analysis if running results are detected.

Output: most important sensors rank by importance (optionally by fault type)

### Module 2: Sensor Inaccuracy Analysis
Sub-Class Name: sensor_inaccuracy_analysis

Inputs: `Monte_Carlo_runs` defines number of runs for Monte Carlo simulation. `rerun=False` defines whether rerun the analysis if running results are detected.

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

`$example_object.sensor_selection_analysis(feature_extraction_config=[['mean', 'std', 'skew', 'kurtosis'], 4*12],
                                         feature_selection_config={'filter': [False, 0.0], 'wrapper': False, 'embedded': True},
                                         by_fault_type=True,
                                         top_n_features=10,
                                         rerun=False)`

The example result is shown as follows:

![Module 1 Results Example](https://github.com/NREL/sensor-impact-evaluation-and-verification/blob/main/figures/module_1_results_example.png)
<!-- <img src="https://github.com/NREL/sensor-impact-evaluation-and-verification/blob/main/figures/module_1_results_example.png" width="600" height="600"> -->

The second analysis is sensor inaccruacy impact on sensor selection and FDD performance. The key input to this module is sensor fault probability table to define the probability of sensor with faults.

`$example_object.sensor_inaccuracy_analysis(Monte_Carlo_runs=1000, rerun=True)`

The example result is shown as follows:

<!-- ![Module 2 Results Example](https://github.com/NREL/sensor-impact-evaluation-and-verification/blob/main/figures/module_2_results_example.png) -->
<img src="https://github.com/NREL/sensor-impact-evaluation-and-verification/blob/main/figures/module_2_results_example.png" width="600" height="600">

The third analysis is sensor cost analysis. The key inputs to be defined in this module is baseline sensor set, candidate sensor set, and analysis mode which defines whether sensors are evaluate one by one or group by group.

`$example_object.sensor_cost_analysis(analysis_mode='group',
                                    baseline_sensor_set='default',
                                    candidate_sensor_set='default',
                                    objective_function_coefficients=[0.11, 150, 15643],
                                    rerun=True)`

The example result is shown as follows:  

<!-- ![Module 3 Results Example](https://github.com/NREL/sensor-impact-evaluation-and-verification/blob/main/figures/module_3_results_example.png) -->
<img src="https://github.com/NREL/sensor-impact-evaluation-and-verification/blob/main/figures/module_3_results_example.png" width="600" height="200">

## Machine Learning Algorithms Options in this module

[Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier)

[Gradient Boosting for classification](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier)

## Unit Test and Integration Test

The codes for the automation of unit test and integration test are located here: `sensor-impact-evaluation-and-verification/test/`

Unit test: test for all three analysis modules using "unittest" module in Python to (1) make sure output matches expectedv values, (2) consider parameter type error

The integration test focuses on the connections between the model setup and each analysis module. We created seven test scenarios. Three scenarios test the whole workflow with only one analysis. The other three scenarios test the whole workflow with two analysis conducted. The final test scenario runs all the three analysis modules.

## Tasks Fnished
Find a better structure to select ML algorithm, 2/22/2022

Documentation to describe how to generate simulation data when technical_route is user_defined_analysis, 2/22/2022

Reduce the size of temporary files generated in sensor_inaccuracy_analysis module, 2/25/2022

Improve the computation efficiency of all three modules, 2/30/2022

Use config.json to define case-specific parameters instead of put all of constants in base_functions.py, 2/30/2022

Improve sensor cost analysis using sensor selection module results instead of a simple feature selection, 3/4/2022

Find a server to store simulation data used for "general_guidance" mode, 5/2
  
Write documentation for preparing simulation data in the "User Defined Analysis" mode, 6/15
  
Write doocumentation to prepare config.json, sensor_fault_probability_table, and sensor group information, 6/21

## Future Work
TODO: use fault prevalence probability to calculate STMC ranges

TODO: add data cleaning function to remove sensor with wrong data type and remove data with bad quality
  
TODO: parallel computing of large simulation data and large Monte Carlo runs

TODO: raise a warning based on ML validation error threshold

TODO: add a function to estimate the calculation time and raise a warning

TODO: include the function of filter method and wrapper method in sensor selection module

TODO: better KDE plot for sensor inaccuracy analysis
