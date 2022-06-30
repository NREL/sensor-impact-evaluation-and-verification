import sensor_impact_FDD as SIF
import os

example_object = SIF.SensorImpactFDD(technical_route='user_defined_analysis',
                                     building_type_or_name='small_commercial_building',
                                     ml_algorithm='random_forest',
                                     weather='TN_Knoxville',
                                     root_path=os.getcwd())

# Create local folder structure to store data and analysis result
example_object.create_folder_structure()

# Download simulation data from server
example_object.download_data()

# Run feature selection analysis module
example_object.sensor_selection_analysis(feature_extraction_config=[['mean', 'std'], 4*12],
                                         feature_selection_config={'filter': [False, 0.0], # ['pearson', 0.5]
                                                                   'wrapper': False, #'wrapper': 'forward'
                                                                   'embedded': True},
                                         by_fault_type=False,
                                         top_n_features=10,
                                         rerun=True)

# Run sensor inaccuracy module
example_object.sensor_inaccuracy_analysis(Monte_Carlo_runs=12, rerun=True)

# Run sensor cost analysis module
example_object.sensor_cost_analysis(analysis_mode='single',
                                    baseline_sensor_set='default',
                                    candidate_sensor_set='default',
                                    objective_function_coefficients=[0.11, 150, 15643],
                                    rerun=True)
