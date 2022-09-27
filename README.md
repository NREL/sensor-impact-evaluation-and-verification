# Python Package: Sensor Impact FDD (SIF)
This is an NREL public repository used for the DOE-funded sensor impact evaluation and verification project. Specifically, the function of the repository is to house an analysis framework that facilitates the evaluation of sensor impact (including sensor inaccuracy, sensor selection, and sensor cost analysis) on fault detection and diagnostics (FDD) performance.

Authors: Liang Zhang, Matt Leach, National Renewable Energy Laboratory, January 18, 2021

## Installation
Download and install the latest version of [Conda](https://docs.conda.io/en/latest/) (version 4.4 or above)
Create a new conda environment:

`$ conda create -n <name-of-repository> python=3.8 pip`

`$ conda activate <name-of-repository>`

(If you’re using a version of Conda older than 4.4, you may need to instead use source activate <name-of-repository>.)

Make sure you are using the latest version of pip:

`$ pip install --upgrade pip`

Install the environment needed for this repository:

`$ pip install -e .[dev]`

## Technical Routes: General Guidance and User Defined Analysis
"General Guidance" mode is fully supported by an example dataset, created using an empirically-calibrated building energy model and corresponding fault model library representing Oak Ridge National Laboratory's Flexible Research Platform (FRP) facility. Check the Data section for download information. Users can run in "General Guidance" mode without providing any additional input data. This mode is useful to demonstrate the functionality of the analysis framework.

"User Defined Analysis" mode allows users to apply the analysis framework to a set of user-defined input data. Guidance for generating the necessary simulation data can be found in `documentations/User_Defined_Analysis_Simulation_Data_Generation_Guidance.docx`.

## Data
"General Guidance" mode leverages simulated building fault data from the Flexible Research Platform analysis case study based on the application of a fault library of 22 fault types across weather data from seven different climate zones.
Details of the physics-based fault modeling process are provided in [this paper](https://www.mdpi.com/2075-5309/9/11/233/htm).

If the user specifies "technical_route" as "general_guidance", the analysis framework will automatically download relevant simulation data from the FRP case study. Users only need to prepare their own input datasets when "technical_route" is specified as "user_defined_analysis".The users need to prepare the data themselves only when the “technical_route” is “user_defined_analysis”. Please contact Matt.Leach@nrel.gov if there are any download issues.

## Modules
sensor_impact_FDD.py leverages individual modules to evaluate sensor impact. This repo contains three sub-classes defining the three modules used to evaluate sensor impact for FDD.

### Module 1: Sensor Selection Analysis
Sub-Class Name: sensor_selection_analysis

Reference: Find details for this module in [this paper] (https://www.sciencedirect.com/science/article/pii/S0360132320307071).

Inputs: `feature_extraction_config` defines feature extraction options. `feature_selection_config` defines feature selection options such as filter, wrapper, and hybrid method. `by_fault_type` defines whether the analysis is conducted for each fault fault type separately or considering all fault types as a group. `top_n_features` defines the number of features (ordered from most to least important) to be included in results `rerun=False` defines whether to rerun the analysis if previous results are detected.

Output: a list of the most important sensors (those with the highest impact on FDD performance), ordered by importance (by individual fault type or across all fault types)

### Module 2: Sensor Inaccuracy Analysis
Sub-Class Name: sensor_inaccuracy_analysis

Reference: Find details for this module in [this paper] (https://link.springer.com/article/10.1007/s12273-021-0833-4).

Inputs: `Monte_Carlo_runs` defines the number of runs for Monte Carlo simulation. `rerun=False` defines whether to rerun the analysis if previous results are detected.

Output: a ranking of sensor selection likelihood (from highest to lowest) given assumed sensor inaccuracy (by individual fault type or across all fault types) and a FDD accuracy Kernel density estimation (KDE) plot which quantifies how individual sensor inaccuracies propagate to impact overall FDD performance.

### Module 3: Sensor Cost Analysis
Sub-Class Name: sensor_cost_analysis

Reference: A paper detailing this module has been accepted by *Energy*; a pre-publication manuscript can be found in 'documentations/Sensor Cost-Effectiveness Analysis for Data-Driven Fault Detection and Diagnostics in Commercial Buildings - Pre-Publication.doc' 

Inputs: `analysis_mode` defines whether the analysis is sensor-specific ('single') or based on a group of sensors ('group'). `baseline_sensor_set` defines the set of sensors that defines the starting point for analysis and comparison; this could be an existing sensor set for an operational building or a proposed design set for a new project. `candidate_sensor_set` defines the proposed sensor set to be evaluated against the baseline set. `objective_function_coefficients` defines (in order) the electricity price ($/kWh), dollar value of thermal comfort, and dollar value of a maintenance visit. `rerun` defines whether to rerun the analysis if previous results are detected.

Output: summary table ranking sensor opportunity values (Sensor Threshold Marginal Cost, or STMC) from highest to lowest, with disaggregated STMC component values for energy efficiency, thermal comfort, and maintenance.

## Example Use Case
Start by creating an object to define the general analysis configuration. In this example, the technical route is "general_guidance", which means that results will be based on the FRP case study rather than a user-specified buildings that the analysis is based on providing general guidance and analysis instead of analyzing specific building.

<pre>
import sensor_impact_FDD as SIF
import base_functions as bf
import os
example_object = SIF.SensorImpactFDD(technical_route='general_guidance',
                                     building_type_or_name='small_commercial_building',
                                     ml_algorithm='random_forest',
                                     weather='TN_Knoxville',
                                     root_path=os.getcwd())
</pre>

After creating the configuration object, three types of analysis can be performed. The first analysis type is sensor selection analysis, which can be used to identify sensors important to FDD performance. There are two forms that this analysis can take; (1) evaluating sensor importance individually by fault type, and (2) evaluating sensor importance overall across all fault types collectively.

<pre>
example_object.sensor_selection_analysis(feature_extraction_config=[['mean', 'std'], 4*12],
                                         feature_selection_config={'filter': [False, 0.0], 'wrapper': False, 'embedded': True},
                                         by_fault_type=True,
                                         top_n_features=10,
                                         rerun=False)
</pre>

The example sensor selection result is as follows:

![Module 1 Results Example](https://github.com/NREL/sensor-impact-evaluation-and-verification/blob/main/figures/module_1_results_example.png)
<!-- <img src="https://github.com/NREL/sensor-impact-evaluation-and-verification/blob/main/figures/module_1_results_example.png" width="600" height="600"> -->

The second analysis type is evaluating the impact of sensor inaccuracy on sensor selection and FDD performance. The key input to this module is the sensor fault probability table that defines the likelihood of a sensor experiencing faulty behavior. The more likely a sensor is to fail, the less likely it would be selected.

`example_object.sensor_inaccuracy_analysis(Monte_Carlo_runs=10, rerun=True)`

The example sernsor inaccuracy analysis result is as follows:

<!-- ![Module 2 Results Example](https://github.com/NREL/sensor-impact-evaluation-and-verification/blob/main/figures/module_2_results_example.png) -->
<img src="https://github.com/NREL/sensor-impact-evaluation-and-verification/blob/main/figures/module_2_results_example.png" width="600" height="600">

The third analysis type is sensor cost analysis. The key inputs to be defined for this module are the baseline sensor set, candidate sensor set, and analysis mode (which defines whether sensors are evaluated individually or as a combined set). Results are based on a 3-year analysis period.


<pre>
example_object.sensor_cost_analysis(analysis_mode='single',
                                    baseline_sensor_set='default',
                                    candidate_sensor_set='default',
                                    objective_function_coefficients=[0.11, 150, 15643],
                                    rerun=True)
</pre>

The example result is shown as follows:  

<!-- ![Module 3 Results Example](https://github.com/NREL/sensor-impact-evaluation-and-verification/blob/main/figures/module_3_results_example.png) -->
<img src="https://github.com/NREL/sensor-impact-evaluation-and-verification/blob/main/figures/module_3_results_example.png" width="600" height="200">

## Machine Learning Algorithms Options supported by this analysis framework

[Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier)

[Gradient Boosting for Classification](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier)

## Unit Tests and Integration Tests

The code that facilitates automated unit testing and integration testing is located here: `sensor-impact-evaluation-and-verification/test/`

Unit testing can be executed for all three analysis modules using the "unittest" module in Python. Unit testing (1) makes sure output matches expected values, (2) considers parameter type error

Integration tests focus on the connections between the model setup and the analysis modules. We created seven integration test scenarios. Three scenarios test the workflow with a single analysis module applied. Three scenarios test the workflow with two analis modules applied. The final scenario tests the workflow with all three analysis modules applied.

