import pandas as pd
import numpy as np
import os
import shutil
import warnings
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

import base_functions

class SensorImpactFDD:
    def __init__(self, technical_route = 'general_guidance', building_type_or_name = 'small_commercial_building',
                 ml_algorithm = 'random_forest', weather = 'TN_Knoxville', root_path = os.getcwd()):
        """

        :param technical_route: choose from 'general_guidance' and 'user_defined_analysis'.
        :param building_type_or_name: name of the target building. If technical route is 'general_guidance',
        the building_type_or_name can only be selected from the keys in guidance_bldg_dict in base_functions
        :param ml_algorithm: machine learning algorithm used for the entire analysis.
        :param weather: the weather that the simulation data is simulated under.
        :param root_path: the path that creates folder structure to save data and results. Unless defined, the default
        value is the directory where the current Python script is located.
        """

        if technical_route not in ['general_guidance', 'user_defined_analysis']:
            raise Exception(f"The value of technical_route should be either 'general_guidance' or "
                            "'user_defined_analysis'. Now its value is {technical_route}")
        else:
            self.technical_route = technical_route


        if (technical_route == 'general_guidance') and (building_type_or_name not in\
                base_functions.Base.guidance_bldg_dict.keys()):
            raise Exception(f"In the general_guidance mode, building_type_or_name should be limited to"
                            f"the values in {list(base_functions.base.guidance_bldg_dict.keys())}."
                            f"Now the value is {building_type_or_name}")
        else:
            self.building_type_or_name = building_type_or_name

        if ml_algorithm not in base_functions.Base.algorithm_dict.keys():
            raise Exception(f"ml_algorithm should be limited to the values in"
                            f" {list(base_functions.base.algorithm_dict.keys())}."
                            f"Now the value is {ml_algorithm}")
        else:
            self.ml_algorithm = ml_algorithm

        self.weather = weather

        self.root_path = root_path

        # TODO: in this version we don't consider control algorithm a variable in the analysis
        # self.ctrl_algorithm = ctrl_algorithm

    def create_folder_structure(self):
        print('Creating folder structure...')
        folders = [#TODO: decide whether to include a folder to save models
                   f'models/{self.technical_route}/{self.building_type_or_name}/{self.weather}/',
                   f'analysis_results/{self.technical_route}/{self.building_type_or_name}/{self.weather}/']

        analysis_types = ['sensor_selection_analysis/',
                         'sensor_inaccuracy_analysis/',
                         'sensor_cost_analysis/']

        all_folder_list = []
        for folder in folders:
            for analysis_type in analysis_types:
                all_folder_list.append(os.path.join(self.root_path, folder + analysis_type))

        self.simulation_data_dir = os.path.join(self.root_path, f'simulation_data/{self.technical_route}/'
                                                            f'{self.building_type_or_name}/{self.weather}/')
        all_folder_list.append(self.simulation_data_dir)

        for folder_name in all_folder_list:
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
        print('Complete!')

    def download_data(self):
        print('Downloading simulation data...')
        # Check whether it is already downloaded first
        if len(os.listdir(self.simulation_data_dir)) == 0:
            print(f'Simulation data folder {self.simulation_data_dir} is empty. Downloading simulation data from '
                  f'server...')
            # TODO: for now the server is not available and the file is copied from a local file
            src = 'C:/Users/leonz/jupyter_notebook/sensor_impact_FDD_framework/data/default_control/TN_Knoxville/'
            dest = self.simulation_data_dir
            src_files = os.listdir(src)
            for file_name in src_files:
                full_file_name = os.path.join(src, file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, dest)
            print(f'Complete!')

        else:
            print(f'Files detected in the simulation data folder {self.simulation_data_dir}')
            # TODO: here we should plugin a simulation data quality detection module
            pass

    #def sensor_config(self):
    #    this

    def sensor_selection_analysis(self, feature_extraction_config, feature_selection_config, by_fault_type,
                                  top_n_features, rerun=False):
        print('-----Sensor Selection Analysis-----')
        print('Results detected.')
        pass

    def sensor_inaccuracy_analysis(self, sensor_fault_probability_table, by_fault_type, top_n_features,
                                   Monte_Carlo_runs, rerun=False):
        print('-----Sensor Inaccuracy Analysis-----')
        print('Results detected.')
        pass

    def sensor_cost_analysis(self, analysis_mode, baseline_sensor_set=None, candidate_sensor_set=None, rerun=False):
        warnings.filterwarnings('ignore')
        # constants
        simulation_metadata_name = 'summary_results_algorithm_Knoxville_TN_AMY.csv'
        sensor_group_info_dir = 'data/metadata/Suggested Sensor Sets.csv'
        sensor_cost_analysis_result_folder_dir = os.path.join(self.root_path,
                                                              f'analysis_results/{self.technical_route}/{self.building_type_or_name}/{self.weather}/'
                                                              f'sensor_cost_analysis/')

        weight_of_baseline_data = 1
        energy_difference_label = True
        training_error_or_testing_error = 'testing_error'

        if analysis_mode == 'single':
            print('-----Sensor Cost Analysis: Single Sensor Mode-----')
            # Read sensor group information
            sensor_group_df = pd.read_csv(sensor_group_info_dir)[['E+ Output Field', 'Sensor Set']].iloc[1:185]

            # The core function to calculate misdetection rate and false alarm rate from fault type and sensor used
            def misdetection_false_alarm_rate(fault_type, sensor_list):
                meta_data = pd.read_csv(f'{self.simulation_data_dir}/{simulation_metadata_name}')
                baseline_file_id = meta_data.loc[meta_data.fault_type == 'baseline'].id.iloc[0]
                baseline_data = pd.read_csv(f'{self.simulation_data_dir}/{baseline_file_id}_sensors.csv')
                baseline_data = baseline_data.groupby(baseline_data.index // 4).mean().iloc[:, 0:-8]
                baseline_data['label'] = 'baseline'

                simulation_file_list = meta_data.loc[meta_data.fault_type == fault_type].id.tolist()
                fault_inputs_output = pd.DataFrame([])
                for simulation_id, i in zip(simulation_file_list, range(len(simulation_file_list))):
                    #         print(f'    Processing: {i+1}/{len(simulation_file_list)}')
                    temp_raw_FDD_data = pd.read_csv(f'data/default_control/TN_Knoxville/{simulation_id}_sensors.csv')
                    temp_raw_FDD_data = temp_raw_FDD_data.groupby(temp_raw_FDD_data.index // (4)).mean().iloc[:, 0:-8]
                    temp_raw_FDD_data['label'] = fault_type
                    fault_inputs_output = pd.concat([fault_inputs_output, temp_raw_FDD_data], axis=0)

                # faulty_and_unfaulty_inputs_output = pd.concat(
                #     [fault_inputs_output, pd.concat([baseline_data] * len(simulation_file_list) * weight_of_baseline_data)],
                #     axis=0).reset_index(
                #     drop=True)

                faulty_and_unfaulty_inputs_output = pd.concat(
                    [fault_inputs_output, pd.concat([baseline_data] * weight_of_baseline_data)],
                    axis=0).reset_index(drop=True)

                if energy_difference_label:
                    electricity_gas_label_df = faulty_and_unfaulty_inputs_output[
                        ['electricity_facility [W]', 'gas_facility [W]',
                         'label']]
                    baseline_electricity = electricity_gas_label_df.loc[electricity_gas_label_df.label == 'baseline'][
                        'electricity_facility [W]']
                    baseline_electricity_repeated = pd.concat(
                        [baseline_electricity] * int(len(electricity_gas_label_df) / len(baseline_electricity)),
                        ignore_index=True)
                    baseline_gas = electricity_gas_label_df.loc[electricity_gas_label_df.label == 'baseline'][
                        'gas_facility [W]']
                    baseline_gas_repeated = pd.concat(
                        [baseline_gas] * int(len(electricity_gas_label_df) / len(baseline_gas)),
                        ignore_index=True)
                    electricity_gas_label_df['baseline_electricity_facility [W]'] = baseline_electricity_repeated
                    electricity_gas_label_df['baseline_gas_facility [W]'] = baseline_gas_repeated
                    electricity_gas_label_df['electricity_over_threshold'] = (abs(
                        electricity_gas_label_df['electricity_facility [W]'] - electricity_gas_label_df[
                            'baseline_electricity_facility [W]']) / electricity_gas_label_df[
                                                                                  'baseline_electricity_facility [W]']) > 0.05
                    electricity_gas_label_df['gas_over_threshold'] = (abs(
                        electricity_gas_label_df['gas_facility [W]'] - electricity_gas_label_df[
                            'baseline_gas_facility [W]']) /
                                                                      electricity_gas_label_df[
                                                                          'baseline_gas_facility [W]']) > 0.05
                    electricity_gas_label_df['electricity_over_threshold_or_gas_over_threshold'] = [x or y for x, y in
                                                                                                    zip(
                                                                                                        electricity_gas_label_df[
                                                                                                            'electricity_over_threshold'],
                                                                                                        electricity_gas_label_df[
                                                                                                            'gas_over_threshold'])]
                    electricity_gas_label_df['adjusted_label'] = electricity_gas_label_df['label'] * \
                                                                 electricity_gas_label_df[
                                                                     'electricity_over_threshold_or_gas_over_threshold']
                    electricity_gas_label_df['adjusted_label'] = electricity_gas_label_df['adjusted_label'].replace('',
                                                                                                                    'baseline')
                    faulty_and_unfaulty_inputs_output['label'] = electricity_gas_label_df['adjusted_label'].rename(
                        'label')

                # model with defined sensors
                inputs = faulty_and_unfaulty_inputs_output[sensor_list]
                output = faulty_and_unfaulty_inputs_output['label']

                cv = KFold(n_splits=4, shuffle=True, random_state=42)
                for train_index, test_index in cv.split(inputs):
                    break

                if training_error_or_testing_error == 'testing_error':
                    X_train, X_test = inputs.iloc[train_index].reset_index(drop=True), inputs.iloc[
                        test_index].reset_index(drop=True)
                    y_train, y_test = output.iloc[train_index].reset_index(drop=True), output.iloc[
                        test_index].reset_index(drop=True)

                if training_error_or_testing_error == 'training_error':
                    X_train, X_test = inputs, inputs
                    y_train, y_test = output, output

                model_with_baseline_sensors = RandomForestClassifier(n_estimators=20, random_state=42)
                #     model_with_baseline_sensors = GradientBoostingClassifier(n_estimators = 20, random_state=42)

                model_with_baseline_sensors.fit(X_train, y_train)
                #     feature_importance_temp = pd.DataFrame([])
                #     feature_importance_temp.loc[:,'sensor_name'] = baseline_sensor_list
                #     feature_importance_temp.loc[:,'importance'] = model_with_baseline_sensors.feature_importances_
                #     important_features += feature_importance_temp.sort_values(
                #         by=['importance'], ascending = False).sensor_name[0:40].tolist()
                # Generate predictions on the test data and collect
                y_test_predicted = model_with_baseline_sensors.predict(X_test)

                temp_df = pd.concat([y_test, pd.Series(y_test_predicted)], axis=1)
                temp_df.columns = ['true_y', 'predicted_y']

                misdetection_rate = len(
                    temp_df.loc[(temp_df.true_y != 'baseline') & (temp_df.predicted_y == 'baseline')]) / len(
                    temp_df)
                false_alarm_rate = len(
                    temp_df.loc[(temp_df.true_y == 'baseline') & (temp_df.predicted_y != 'baseline')]) / len(
                    temp_df)

                return (misdetection_rate, false_alarm_rate,)

            # iterate over single sensor
            # fault_type_list = ['economizer_opening_stuck', 'hvac_setback_error_no_overnight_setback']
            # fault_type_list = top_10_electricity_consumption_faults_list
            # fault_type_list = top_10_site_energy_faults_list[0:10]
            # fault_type_list = ['hvac_setback_error_no_overnight_setback', 'economizer_opening_stuck',
            #                    'air_handling_unit_fan_motor_degradation', 'excessive_infiltration', 'liquid_line_restriction']
            # fault_type_list = ['thermostat_bias', 'economizer_opening_stuck', 'supply_air_duct_leakages'] # focus on thermal comfort improvement
            fault_type_list = ['thermostat_bias', 'economizer_opening_stuck',
                               'lighting_setback_error_no_overnight_setback',
                               'excessive_infiltration',
                               'liquid_line_restriction']  # focus on thermal comfort improvement and electricity improvement

            baseline_sensor_list = sensor_group_df.loc[sensor_group_df['Sensor Set'].isin(['Basic', 'Moderate'])][
                'E+ Output Field'].tolist()
            single_sensor_pool = sensor_group_df.loc[sensor_group_df['Sensor Set'].isin(['Rich'])][
                'E+ Output Field'].tolist()

            result_df = pd.DataFrame([])
            for fault_type in fault_type_list:
                print(f'    Processing: {fault_type}')
                print(f'    Processing: Baseline...')
                error = misdetection_false_alarm_rate(fault_type, baseline_sensor_list)
                result_df = pd.concat([result_df, pd.DataFrame([fault_type, 'Baseline', error[0], error[1]])], axis=1)

                for additional_sensor, j in zip(single_sensor_pool, range(len(single_sensor_pool))):
                    print(f'    Processing: {additional_sensor}, {j + 1}/{len(single_sensor_pool)}...')
                    error = misdetection_false_alarm_rate(fault_type, baseline_sensor_list + [additional_sensor])
                    result_df = pd.concat(
                        [result_df, pd.DataFrame([fault_type, additional_sensor, error[0], error[1]])], axis=1)

            result_df = result_df.transpose()
            result_df.columns = ['Fault_Type', 'Additional_Single_Sensor', 'Misdetection_Rate', 'False_Alarm_Rate']
            result_df = result_df.reset_index(drop=True)

            result_df_raw = result_df.copy()
            result_df_raw.to_csv(f'{sensor_cost_analysis_result_folder_dir}/single_additional_sensor_error_rates.csv',
                                 index=None)

            def average_annual_electricity_kWh(fault_type):
                meta_data = pd.read_csv(
                    'data\default_control\TN_Knoxville\summary_results_algorithm_Knoxville_TN_AMY.csv')

                simulation_file_list = meta_data.loc[meta_data.fault_type == fault_type].id.tolist()

                fault_inputs_output = pd.DataFrame([])

                for simulation_id, i in zip(simulation_file_list, range(len(simulation_file_list))):
                    temp_raw_FDD_data = pd.read_csv(f'data/default_control/TN_Knoxville/{simulation_id}_sensors.csv')
                    temp_raw_FDD_data = temp_raw_FDD_data.groupby(temp_raw_FDD_data.index // (4)).mean().iloc[:, 0:-8]
                    temp_raw_FDD_data['label'] = fault_type

                    fault_inputs_output = pd.concat([fault_inputs_output, temp_raw_FDD_data], axis=0)

                average_annual_electricity_kWh = fault_inputs_output[
                                                     'electricity_facility [W]'].sum() * 3600 / 1000 / 3600 / len(
                    simulation_file_list)

                return average_annual_electricity_kWh

            dict_fault_type_average_annual_electricity_kWh = {}
            for fault_type in fault_type_list:
                average_annual_electricity_kWh_temp = average_annual_electricity_kWh(fault_type)
                dict_fault_type_average_annual_electricity_kWh[fault_type] = average_annual_electricity_kWh_temp

            result_df['annual_electricity_kWh'] = result_df.Fault_Type.map(
                dict_fault_type_average_annual_electricity_kWh)
            result_df['unfaulty_annual_electricity_kWh'] = average_annual_electricity_kWh('baseline')
            result_df['annual_electricity_diff_kWh'] = result_df['annual_electricity_kWh'] - result_df[
                'unfaulty_annual_electricity_kWh']

            df = result_df.loc[result_df.Additional_Single_Sensor == 'Baseline']
            newdf = pd.DataFrame(np.repeat(df.values, len(result_df.Additional_Single_Sensor.unique()), axis=0))
            newdf.columns = df.columns

            result_df['Baseline_Misdetection_Rate'] = newdf['Misdetection_Rate']
            result_df['Baseline_False_Alarm_Rate'] = newdf['False_Alarm_Rate']

            result_df['Misdetection_Rate_Diff'] = result_df['Misdetection_Rate'] - result_df[
                'Baseline_Misdetection_Rate']
            result_df['False_Alarm_Rate_Diff'] = result_df['False_Alarm_Rate'] - result_df['Baseline_False_Alarm_Rate']

            result_df['unmet_hours_during_occupied_cooling_and_heating_diff'] = result_df['Fault_Type'].map(
                base_functions.Base.thermal_discomfort_dict)

            # https://www.kub.org/uploads/GSA_45.pdf
            electricity_cost = 0.11  # USD/kWh
            C1 = 150
            C2 = 15643

            result_df['benefit_from_less_energy_wasted_by_faults_USD'] = -electricity_cost * result_df[
                'Misdetection_Rate_Diff'] \
                                                                         * result_df['annual_electricity_diff_kWh']
            temp_df = result_df['benefit_from_less_energy_wasted_by_faults_USD'].copy()
            temp_df[temp_df < 0] = 0
            result_df['benefit_from_less_energy_wasted_by_faults_USD_adjusted'] = temp_df

            result_df['benefit_from_less_maintenance_USD'] = - C2 * result_df['False_Alarm_Rate_Diff']
            temp_df = result_df['benefit_from_less_maintenance_USD'].copy()
            temp_df[temp_df < 0] = 0
            result_df['benefit_from_less_maintenance_USD_adjusted'] = temp_df

            result_df['benefit_from_thermal_comfort_USD'] = - C1 * result_df[
                'Misdetection_Rate_Diff'] * result_df['unmet_hours_during_occupied_cooling_and_heating_diff']
            temp_df = result_df['benefit_from_thermal_comfort_USD'].copy()
            temp_df[temp_df < 0] = 0
            result_df['benefit_from_thermal_comfort_USD_adjusted'] = temp_df

            # result_df['total_benefit_USD'] = -electricity_cost * result_df['Misdetection_Rate_Diff'] * result_df[
            #     'annual_electricity_diff_kWh'] - C2 * result_df['False_Alarm_Rate_Diff'] - C1 * result_df[
            #     'Misdetection_Rate_Diff'] * result_df['unmet_hours_during_occupied_cooling_and_heating_diff']

            result_df['total_benefit_USD'] = result_df['benefit_from_less_energy_wasted_by_faults_USD_adjusted'] + \
                                             result_df[
                                                 'benefit_from_less_maintenance_USD_adjusted'] + result_df[
                                                 'benefit_from_thermal_comfort_USD_adjusted']

            result_df.to_csv(f'{sensor_cost_analysis_result_folder_dir}/single_additional_sensor_processed.csv',
                             index=None)

            # result_df = pd.read_csv(f'{sensor_cost_analysis_result_folder_dir}/single_additional_sensor_processed.csv')

            # Introducing fault prevalence
            result_df['fault_prevalence'] = result_df['Fault_Type'].map(base_functions.Base.fault_prevalence_dict)

            result_df['energy_improvement_benefit_USD_weighted_by_fault_prevalence'] = result_df[
                                                                                           'benefit_from_less_energy_wasted_by_faults_USD_adjusted'] * \
                                                                                       result_df[
                                                                                           'fault_prevalence']

            result_df['comfort_improvement_benefit_USD_weighted_by_fault_prevalence'] = result_df[
                                                                                            'benefit_from_less_maintenance_USD_adjusted'] * \
                                                                                        result_df[
                                                                                            'fault_prevalence']

            result_df['maintenace_improvement_benefit_USD_weighted_by_fault_prevalence'] = result_df[
                                                                                               'benefit_from_thermal_comfort_USD_adjusted'] * \
                                                                                           result_df[
                                                                                               'fault_prevalence']

            result_df['total_benefit_USD_weighted_by_fault_prevalence'] = result_df['total_benefit_USD'] * result_df[
                'fault_prevalence']

            result_df.iloc[:, 2:] = result_df.iloc[:, 2:].astype('float')
            result_df[result_df.total_benefit_USD_weighted_by_fault_prevalence <
                      0].total_benefit_USD_weighted_by_fault_prevalence = 0
            result_df_fault_prevalence = result_df.groupby(['Additional_Single_Sensor']).sum()[[
                'energy_improvement_benefit_USD_weighted_by_fault_prevalence',
                'comfort_improvement_benefit_USD_weighted_by_fault_prevalence',
                'maintenace_improvement_benefit_USD_weighted_by_fault_prevalence',
                'total_benefit_USD_weighted_by_fault_prevalence']]
            result_df_fault_prevalence.to_csv(
                f'{sensor_cost_analysis_result_folder_dir}/single_additional_sensor_processed_fault_prevalence.csv')


        elif analysis_mode == 'group':
            # Read sensor group information
            sensor_group_df = pd.read_csv(sensor_group_info_dir)[['E+ Output Field', 'Sensor Set']].iloc[1:185]

            baseline_sensor_list = sensor_group_df.loc[sensor_group_df['Sensor Set'].isin(['Basic', 'Moderate'])][
                'E+ Output Field'].tolist()

            def average_annual_electricity_kWh(fault_type):
                meta_data = pd.read_csv('data\default_control\TN_Knoxville\summary_results_algorithm_Knoxville_TN_AMY.csv')

                simulation_file_list = meta_data.loc[meta_data.fault_type == fault_type].id.tolist()

                fault_inputs_output = pd.DataFrame([])

                for simulation_id, i in zip(simulation_file_list, range(len(simulation_file_list))):
                    temp_raw_FDD_data = pd.read_csv(f'data/default_control/TN_Knoxville/{simulation_id}_sensors.csv')
                    temp_raw_FDD_data = temp_raw_FDD_data.groupby(temp_raw_FDD_data.index // (4)).mean().iloc[:, 0:-8]
                    temp_raw_FDD_data['label'] = fault_type

                    fault_inputs_output = pd.concat([fault_inputs_output, temp_raw_FDD_data], axis=0)

                average_annual_electricity_kWh = fault_inputs_output[
                                                     'electricity_facility [W]'].sum() * 3600 / 1000 / 3600 / len(
                    simulation_file_list)

                return average_annual_electricity_kWh

            def misdetection_false_alarm_rate(fault_type, sensor_list):
                meta_data = pd.read_csv(f'{self.simulation_data_dir}/{simulation_metadata_name}')
                baseline_file_id = meta_data.loc[meta_data.fault_type == 'baseline'].id.iloc[0]
                baseline_data = pd.read_csv(f'{self.simulation_data_dir}/{baseline_file_id}_sensors.csv')
                baseline_data = baseline_data.groupby(baseline_data.index // 4).mean().iloc[:, 0:-8]
                baseline_data['label'] = 'baseline'

                simulation_file_list = meta_data.loc[meta_data.fault_type == fault_type].id.tolist()
                fault_inputs_output = pd.DataFrame([])
                for simulation_id, i in zip(simulation_file_list, range(len(simulation_file_list))):
                    #         print(f'    Processing: {i+1}/{len(simulation_file_list)}')
                    temp_raw_FDD_data = pd.read_csv(f'data/default_control/TN_Knoxville/{simulation_id}_sensors.csv')
                    temp_raw_FDD_data = temp_raw_FDD_data.groupby(temp_raw_FDD_data.index // (4)).mean().iloc[:, 0:-8]
                    temp_raw_FDD_data['label'] = fault_type
                    fault_inputs_output = pd.concat([fault_inputs_output, temp_raw_FDD_data], axis=0)

                # faulty_and_unfaulty_inputs_output = pd.concat(
                #     [fault_inputs_output, pd.concat([baseline_data] * len(simulation_file_list) * weight_of_baseline_data)],
                #     axis=0).reset_index(
                #     drop=True)

                faulty_and_unfaulty_inputs_output = pd.concat(
                    [fault_inputs_output, pd.concat([baseline_data] * weight_of_baseline_data)],
                    axis=0).reset_index(drop=True)

                if energy_difference_label:
                    electricity_gas_label_df = faulty_and_unfaulty_inputs_output[
                        ['electricity_facility [W]', 'gas_facility [W]',
                         'label']]
                    baseline_electricity = electricity_gas_label_df.loc[electricity_gas_label_df.label == 'baseline'][
                        'electricity_facility [W]']
                    baseline_electricity_repeated = pd.concat(
                        [baseline_electricity] * int(len(electricity_gas_label_df) / len(baseline_electricity)),
                        ignore_index=True)
                    baseline_gas = electricity_gas_label_df.loc[electricity_gas_label_df.label == 'baseline'][
                        'gas_facility [W]']
                    baseline_gas_repeated = pd.concat(
                        [baseline_gas] * int(len(electricity_gas_label_df) / len(baseline_gas)),
                        ignore_index=True)
                    electricity_gas_label_df['baseline_electricity_facility [W]'] = baseline_electricity_repeated
                    electricity_gas_label_df['baseline_gas_facility [W]'] = baseline_gas_repeated
                    electricity_gas_label_df['electricity_over_threshold'] = (abs(
                        electricity_gas_label_df['electricity_facility [W]'] - electricity_gas_label_df[
                            'baseline_electricity_facility [W]']) / electricity_gas_label_df[
                                                                                  'baseline_electricity_facility [W]']) > 0.05
                    electricity_gas_label_df['gas_over_threshold'] = (abs(
                        electricity_gas_label_df['gas_facility [W]'] - electricity_gas_label_df[
                            'baseline_gas_facility [W]']) /
                                                                      electricity_gas_label_df[
                                                                          'baseline_gas_facility [W]']) > 0.05
                    electricity_gas_label_df['electricity_over_threshold_or_gas_over_threshold'] = [x or y for x, y in zip(
                        electricity_gas_label_df['electricity_over_threshold'],
                        electricity_gas_label_df['gas_over_threshold'])]
                    electricity_gas_label_df['adjusted_label'] = electricity_gas_label_df['label'] * \
                                                                 electricity_gas_label_df[
                                                                     'electricity_over_threshold_or_gas_over_threshold']
                    electricity_gas_label_df['adjusted_label'] = electricity_gas_label_df['adjusted_label'].replace('',
                                                                                                                    'baseline')
                    faulty_and_unfaulty_inputs_output['label'] = electricity_gas_label_df['adjusted_label'].rename('label')

                # model with defined sensors
                inputs = faulty_and_unfaulty_inputs_output[sensor_list]
                output = faulty_and_unfaulty_inputs_output['label']

                cv = KFold(n_splits=4, shuffle=True, random_state=42)
                for train_index, test_index in cv.split(inputs):
                    break

                if training_error_or_testing_error == 'testing_error':
                    X_train, X_test = inputs.iloc[train_index].reset_index(drop=True), inputs.iloc[test_index].reset_index(
                        drop=True)
                    y_train, y_test = output.iloc[train_index].reset_index(drop=True), output.iloc[test_index].reset_index(
                        drop=True)

                if training_error_or_testing_error == 'training_error':
                    X_train, X_test = inputs, inputs
                    y_train, y_test = output, output

                model_with_baseline_sensors = RandomForestClassifier(n_estimators=20, random_state=42)
                #     model_with_baseline_sensors = GradientBoostingClassifier(n_estimators = 20, random_state=42)

                model_with_baseline_sensors.fit(X_train, y_train)
                #     feature_importance_temp = pd.DataFrame([])
                #     feature_importance_temp.loc[:,'sensor_name'] = baseline_sensor_list
                #     feature_importance_temp.loc[:,'importance'] = model_with_baseline_sensors.feature_importances_
                #     important_features += feature_importance_temp.sort_values(
                #         by=['importance'], ascending = False).sensor_name[0:40].tolist()
                # Generate predictions on the test data and collect
                y_test_predicted = model_with_baseline_sensors.predict(X_test)

                temp_df = pd.concat([y_test, pd.Series(y_test_predicted)], axis=1)
                temp_df.columns = ['true_y', 'predicted_y']

                misdetection_rate = len(
                    temp_df.loc[(temp_df.true_y != 'baseline') & (temp_df.predicted_y == 'baseline')]) / len(
                    temp_df)
                false_alarm_rate = len(
                    temp_df.loc[(temp_df.true_y == 'baseline') & (temp_df.predicted_y != 'baseline')]) / len(
                    temp_df)

                return (misdetection_rate, false_alarm_rate,)

            # Sensor Group
            print('-----Sensor Cost Analysis: Sensor Group Mode-----')

            important_sensors_df = pd.DataFrame([])
            # TODO: delete[0:2]
            fault_type_list = ['thermostat_bias', 'economizer_opening_stuck', 'lighting_setback_error_no_overnight_setback',
                               'excessive_infiltration',
                               'liquid_line_restriction'][0:2]  # focus on thermal comfort improvement and electricity
                # improvement

            print(f'    Identifying important sensors for each fault...')
            for fault_type in fault_type_list:

                print(f'        Processing: {fault_type}')

                meta_data = pd.read_csv(f'{self.simulation_data_dir}/{simulation_metadata_name}')

                baseline_file_id = meta_data.loc[meta_data.fault_type == 'baseline'].id.iloc[0]
                baseline_data = pd.read_csv(f'{self.simulation_data_dir}/{baseline_file_id}_sensors.csv')
                baseline_data = baseline_data.groupby(baseline_data.index // 4).mean().iloc[:, 0:-8]
                baseline_data['label'] = 'baseline'

                simulation_file_list = meta_data.loc[meta_data.fault_type == fault_type].id.tolist()
                fault_inputs_output = pd.DataFrame([])
                for simulation_id, i in zip(simulation_file_list, range(len(simulation_file_list))):
                    temp_raw_FDD_data = pd.read_csv(f'{self.simulation_data_dir}/{simulation_id}_sensors.csv')
                    temp_raw_FDD_data = temp_raw_FDD_data.groupby(temp_raw_FDD_data.index // (4)).mean().iloc[:, 0:-8]
                    temp_raw_FDD_data['label'] = fault_type

                    fault_inputs_output = pd.concat([fault_inputs_output, temp_raw_FDD_data], axis=0)

                faulty_and_unfaulty_inputs_output = pd.concat(
                    [fault_inputs_output, pd.concat([baseline_data] * len(simulation_file_list))], axis=0).reset_index(
                    drop=True)

                # model with defined sensors
                inputs = faulty_and_unfaulty_inputs_output.iloc[:, 0:-1]
                output = faulty_and_unfaulty_inputs_output['label']

                cv = KFold(n_splits=4, shuffle=True, random_state=42)
                for train_index, test_index in cv.split(inputs):
                    break

                X_train, X_test = inputs.iloc[train_index].reset_index(drop=True), inputs.iloc[test_index].reset_index(
                    drop=True)
                y_train, y_test = output.iloc[train_index].reset_index(drop=True), output.iloc[test_index].reset_index(
                    drop=True)

                model_with_baseline_sensors = RandomForestClassifier(n_estimators=25, random_state=42)
                #     model_with_baseline_sensors = GradientBoostingClassifier(n_estimators = 20, random_state=42)

                model_with_baseline_sensors.fit(X_train, y_train)

                feature_importance_temp = pd.DataFrame([])
                feature_importance_temp.loc[:, 'sensor_name'] = inputs.columns
                feature_importance_temp.loc[:, 'importance'] = model_with_baseline_sensors.feature_importances_
                important_features = feature_importance_temp.sort_values(
                    by=['importance'], ascending=False).sensor_name.tolist()
                important_features_without_moderate_and_basic_sensor = [x for x in important_features if
                                                                        x not in baseline_sensor_list]

                important_sensors_df[fault_type] = important_features_without_moderate_and_basic_sensor

            important_sensors_df.to_csv(
                f'{sensor_cost_analysis_result_folder_dir}/important_sensor_list_for_all_faults.csv')

            result_multiple_sensor_df = pd.DataFrame([])
            print(f'    Cost analysis by group...')
            for fault_type in fault_type_list:
                print(f'        Processing: {fault_type}')
                # print(f'      Processing: Baseline...')
                error = misdetection_false_alarm_rate(fault_type, baseline_sensor_list)
                result_multiple_sensor_df = pd.concat(
                    [result_multiple_sensor_df, pd.DataFrame([fault_type, 'Baseline', error[0], error[1]])], axis=1)

                # for additional_sensor_number in [5, 10, 15, 20]:
                for additional_sensor_number in [3, 5, 10, 20]:
                    additional_sensor = important_sensors_df[fault_type].iloc[0:additional_sensor_number].tolist()
                    error = misdetection_false_alarm_rate(fault_type, baseline_sensor_list + additional_sensor)
                    result_multiple_sensor_df = pd.concat(
                        [result_multiple_sensor_df,
                         pd.DataFrame([fault_type, additional_sensor_number, error[0], error[1]])],
                        axis=1)

            result_multiple_sensor_df = result_multiple_sensor_df.transpose()
            result_multiple_sensor_df.columns = ['Fault_Type', 'Number_of_Additional_Sensor', 'Misdetection_Rate',
                                                 'False_Alarm_Rate']
            result_multiple_sensor_df = result_multiple_sensor_df.reset_index(drop=True)

            result_multiple_sensor_df_raw = result_multiple_sensor_df.copy()
            result_multiple_sensor_df_raw.to_csv(f'{sensor_cost_analysis_result_folder_dir}/'
                                                 f'multiple_additional_sensor_error_rates.csv', index=None)

            dict_fault_type_average_annual_electricity_kWh = {}
            for fault_type in fault_type_list:
                average_annual_electricity_kWh_temp = average_annual_electricity_kWh(fault_type)
                dict_fault_type_average_annual_electricity_kWh[fault_type] = average_annual_electricity_kWh_temp

            result_multiple_sensor_df['annual_electricity_kWh'] = result_multiple_sensor_df.Fault_Type.map(
                dict_fault_type_average_annual_electricity_kWh)
            result_multiple_sensor_df['unfaulty_annual_electricity_kWh'] = average_annual_electricity_kWh('baseline')
            result_multiple_sensor_df['annual_electricity_diff_kWh'] = result_multiple_sensor_df['annual_electricity_kWh'] - \
                                                                       result_multiple_sensor_df[
                                                                           'unfaulty_annual_electricity_kWh']

            df = result_multiple_sensor_df.loc[result_multiple_sensor_df.Number_of_Additional_Sensor == 'Baseline']
            newdf = pd.DataFrame(
                np.repeat(df.values, len(result_multiple_sensor_df.Number_of_Additional_Sensor.unique()), axis=0))
            newdf.columns = df.columns

            result_multiple_sensor_df['Baseline_Misdetection_Rate'] = newdf['Misdetection_Rate']
            result_multiple_sensor_df['Baseline_False_Alarm_Rate'] = newdf['False_Alarm_Rate']

            result_multiple_sensor_df['Misdetection_Rate_Diff'] = result_multiple_sensor_df['Misdetection_Rate'] - \
                                                                  result_multiple_sensor_df['Baseline_Misdetection_Rate']
            result_multiple_sensor_df['False_Alarm_Rate_Diff'] = result_multiple_sensor_df['False_Alarm_Rate'] - \
                                                                 result_multiple_sensor_df['Baseline_False_Alarm_Rate']

            result_multiple_sensor_df['unmet_hours_during_occupied_cooling_and_heating_diff'] = result_multiple_sensor_df[
                'Fault_Type'].map(base_functions.Base.thermal_discomfort_dict)

            # https://www.kub.org/uploads/GSA_45.pdf
            electricity_cost = 0.11  # USD/kWh
            C1 = 150
            C2 = 15643

            result_multiple_sensor_df['benefit_from_less_energy_wasted_by_faults_USD'] = -electricity_cost * \
                                                                                         result_multiple_sensor_df[
                                                                                             'Misdetection_Rate_Diff'] \
                                                                                         * result_multiple_sensor_df[
                                                                                             'annual_electricity_diff_kWh']
            temp_df = result_multiple_sensor_df['benefit_from_less_energy_wasted_by_faults_USD'].copy()
            temp_df[temp_df < 0] = 0
            result_multiple_sensor_df['benefit_from_less_energy_wasted_by_faults_USD_adjusted'] = temp_df

            result_multiple_sensor_df['benefit_from_less_maintenance_USD'] = - C2 * result_multiple_sensor_df[
                'False_Alarm_Rate_Diff']
            temp_df = result_multiple_sensor_df['benefit_from_less_maintenance_USD'].copy()
            temp_df[temp_df < 0] = 0
            result_multiple_sensor_df['benefit_from_less_maintenance_USD_adjusted'] = temp_df

            result_multiple_sensor_df['benefit_from_thermal_comfort_USD'] = - C1 * result_multiple_sensor_df[
                'Misdetection_Rate_Diff'] * result_multiple_sensor_df[
                                                                                'unmet_hours_during_occupied_cooling_and_heating_diff']
            temp_df = result_multiple_sensor_df['benefit_from_thermal_comfort_USD'].copy()
            temp_df[temp_df < 0] = 0
            result_multiple_sensor_df['benefit_from_thermal_comfort_USD_adjusted'] = temp_df

            # result_multiple_sensor_df['total_benefit_USD'] = -electricity_cost * result_multiple_sensor_df['Misdetection_Rate_Diff'] * result_multiple_sensor_df[
            #     'annual_electricity_diff_kWh'] - C2 * result_multiple_sensor_df['False_Alarm_Rate_Diff'] - C1 * result_multiple_sensor_df[
            #     'Misdetection_Rate_Diff'] * result_multiple_sensor_df['unmet_hours_during_occupied_cooling_and_heating_diff']

            result_multiple_sensor_df['total_benefit_USD'] = result_multiple_sensor_df[
                                                                 'benefit_from_less_energy_wasted_by_faults_USD_adjusted'] + \
                                                             result_multiple_sensor_df[
                                                                 'benefit_from_less_maintenance_USD_adjusted'] + \
                                                             result_multiple_sensor_df[
                                                                 'benefit_from_thermal_comfort_USD_adjusted']

            result_multiple_sensor_df.to_csv(
                f'{sensor_cost_analysis_result_folder_dir}/multiple_additional_sensor_processed.csv',
                index=None)

            # result_multiple_sensor_df['total_benefit_USD'] = -electricity_cost * result_multiple_sensor_df[
            #     'Misdetection_Rate_Diff'] * result_multiple_sensor_df['annual_electricity_diff_kWh']
            #
            # result_multiple_sensor_df.to_csv(f'{sensor_cost_analysis_result_folder_dir}/multiple_additional_sensor_processed.csv',
            #                                  index=None)

            # Introducing fault prevalence
            result_multiple_sensor_df['fault_prevalence'] = result_multiple_sensor_df['Fault_Type'].map(
                base_functions.Base.fault_prevalence_dict)

            result_multiple_sensor_df['energy_improvement_benefit_USD_weighted_by_fault_prevalence'] = \
            result_multiple_sensor_df[
                'benefit_from_less_energy_wasted_by_faults_USD_adjusted'] * result_multiple_sensor_df[
                'fault_prevalence']

            result_multiple_sensor_df['comfort_improvement_benefit_USD_weighted_by_fault_prevalence'] = \
            result_multiple_sensor_df[
                'benefit_from_less_maintenance_USD_adjusted'] * result_multiple_sensor_df[
                'fault_prevalence']

            result_multiple_sensor_df['maintenace_improvement_benefit_USD_weighted_by_fault_prevalence'] = \
            result_multiple_sensor_df[
                'benefit_from_thermal_comfort_USD_adjusted'] * result_multiple_sensor_df[
                'fault_prevalence']

            result_multiple_sensor_df['total_benefit_USD_weighted_by_fault_prevalence'] = result_multiple_sensor_df[
                                                                                              'total_benefit_USD'] * \
                                                                                          result_multiple_sensor_df[
                                                                                              'fault_prevalence']

            result_multiple_sensor_df.iloc[:, 2:] = result_multiple_sensor_df.iloc[:, 2:].astype('float')
            result_multiple_sensor_df[result_multiple_sensor_df.total_benefit_USD_weighted_by_fault_prevalence <
                                      0].total_benefit_USD_weighted_by_fault_prevalence = 0
            result_multiple_sensor_df_fault_prevalence = \
            result_multiple_sensor_df.groupby(['Number_of_Additional_Sensor']).sum()[[
                'energy_improvement_benefit_USD_weighted_by_fault_prevalence',
                'comfort_improvement_benefit_USD_weighted_by_fault_prevalence',
                'maintenace_improvement_benefit_USD_weighted_by_fault_prevalence',
                'total_benefit_USD_weighted_by_fault_prevalence']]
            result_multiple_sensor_df_fault_prevalence.to_csv(
                f'{sensor_cost_analysis_result_folder_dir}/multiple_additional_sensor_processed_fault_prevalence.csv')
            print('Results generated.')

        else:
            raise Exception(f"The value of analysis_mode should be either 'single' or "
                            "'group'. Now its value is {analysis_mode}")