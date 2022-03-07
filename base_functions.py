import pandas as pd

class Base:
    ctrl_datadir_dict = {
        "default_control": f"data/default_control/TN_Knoxville"
    }

    guidance_bldg_dict = {
        'small_commercial_building': 'data/XX/XX'
    }

    algorithm_dict = {
        'random_forest': 'XXX',
        'SVM': "XXX"
    }

    baseline_sensor_set = ['sensor_1', 'sensor_2','sensor_3','sensor_4','sensor_5']

    candidate_sensor_set = ['sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10']

    sensor_fault_probability_table = 'data/metadata/sensor_fault_probability_table.csv'

    fault_prevalence_dict = {

        'air_handling_unit_fan_motor_degradation': sum([0.16, 0.35])/len([0.16, 0.35]),
        'baseline': 0.05, #default
        'biased_economizer_sensor_mixed_t': sum([0.27, 0.19, 0.11])/len([0.27, 0.19, 0.11]),
        'biased_economizer_sensor_oat': sum([0.27, 0.19, 0.11])/len([0.27, 0.19, 0.11]),
        'biased_economizer_sensor_outdoor_rh': sum([0.27, 0.19, 0.11])/len([0.27, 0.19, 0.11]),
        'biased_economizer_sensor_return_rh': sum([0.27, 0.19, 0.11])/len([0.27, 0.19, 0.11]),
        'biased_economizer_sensor_return_t': sum([0.27, 0.19, 0.11])/len([0.27, 0.19, 0.11]),
        'condenser_fan_degradation': sum([0.16, 0.35])/len([0.16, 0.35]),
        'condenser_fouling': 0,
        'duct_fouling': 0.05, #default
        'economizer_opening_stuck': sum([0.11, 0.24, 0.13])/len([0.11, 0.24, 0.13]),
        'excessive_infiltration': 0.15,
        'hvac_setback_error_delayed_onset': 0.05, #default
        'hvac_setback_error_early_termination': 0.05, #default
        'hvac_setback_error_no_overnight_setback': 0.05, #default
        'improper_time_delay_setting_in_occupancy_sensors': sum([0.27, 0.19, 0.11])/len([0.27, 0.19, 0.11]),
        'lighting_setback_error_delayed_onset': 0.05, #default
        'lighting_setback_error_early_termination': 0.05, #default
        'lighting_setback_error_no_overnight_setback': 0.05, #default
        'liquid_line_restriction': 0.03,
        'non_standard_charging_2': sum([0.37, 0.50, 0.53, 0.46, 0.72])/len([0.37, 0.50, 0.53, 0.46, 0.72]),
        'oversized_equipment_at_design': 0.4,
        'presence_of_non_condensable': 0,
        'return_air_duct_leakages': sum([0.38, 0.65, 0.42, 0.39, 0.44])/len([0.38, 0.65, 0.42, 0.39, 0.44]),
        'supply_air_duct_leakages': sum([0.38, 0.65, 0.42, 0.39, 0.44])/len([0.38, 0.65, 0.42, 0.39, 0.44]),
        'thermostat_bias': sum([0.27, 0.19, 0.11])/len([0.27, 0.19, 0.11])}

    results = pd.read_csv(f'C:/Users/leonz/jupyter_notebook/sensor_impact_FDD_framework/data/default_control'
                          f'/TN_Knoxville/summary_results_algorithm_Knoxville_TN_AMY.csv')

    temp_df = results.groupby(['fault_type']).mean()[['unmet_hours_during_occupied_cooling',
                                                      'unmet_hours_during_occupied_heating']]

    temp_df['unmet_hours_during_occupied_cooling_diff'] = temp_df['unmet_hours_during_occupied_cooling'] - 393.5
    temp_df['unmet_hours_during_occupied_heating_diff'] = temp_df['unmet_hours_during_occupied_heating'] - 443.5

    temp_df['unmet_hours_during_occupied_cooling_and_heating_diff'] = temp_df[
                                                                          'unmet_hours_during_occupied_cooling_diff'] \
                                                                      + temp_df[
                                                                          'unmet_hours_during_occupied_heating_diff']

    temp_df = temp_df['unmet_hours_during_occupied_cooling_and_heating_diff']

    thermal_discomfort_dict = temp_df.to_dict()
