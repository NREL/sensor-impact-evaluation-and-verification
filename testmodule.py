# TODO: find a better structure to select ML algorithm
# TODO: documentation to describe how to generate simulation data when technical_route is user_defined_analysis
# TODO: reduce the size of temporary files generated in sensor_inaccuracy_analysis module
# TODO: improve the computation efficiency of all three modules
# TODO: use config.json to define case-specific parameters instead of put all of constants in base_functions.py
# TODO: improve sensor cost analysis using sensor selection module results instead of a simple feature selection

import sensor_impact_FDD as SIF
import base_functions as bf
import os

example_object = SIF.SensorImpactFDD(technical_route='general_guidance',
                                     building_type_or_name='small_commercial_building',
                                     ml_algorithm='random_forest',
                                     weather='TN_Knoxville',
                                     root_path=os.getcwd())

# Create local folder structure to store data and analysis result
example_object.create_folder_structure()

# Download simulation data from server
example_object.download_data()

# Run feature selection analysis module
example_object.sensor_selection_analysis(feature_extraction_config=[['mean', 'sum', 'std', 'skew', 'kurt'], '4h'],
                                         feature_selection_config=[['filter', 'pearson', 0.5], ['wrapper', 'forward'],
                                                                   ['embedded', False]],
                                         by_fault_type=True,
                                         top_n_features=3,
                                         rerun=False)

# Run sensor inaccuracy module
example_object.sensor_inaccuracy_analysis(sensor_fault_probability_table=bf.Base.sensor_fault_probability_table,
                                          by_fault_type=False,
                                          top_n_features=20,
                                          Monte_Carlo_runs=1000,
                                          rerun=False)

# Run sensor cost analysis module
example_object.sensor_cost_analysis(analysis_mode='group',
                                    baseline_sensor_set=bf.Base.baseline_sensor_set,
                                    candidate_sensor_set=bf.Base.candidate_sensor_set,
                                    rerun=True)
