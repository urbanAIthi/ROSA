import os
import pandas as pd
from typing import List, Union


def evaluate_performance(experiment_name: str,
                         evaluation_scenarios: List[Union[str, int]],
                         metrics: List[str],
                         pre_eval_data: List[List[float]],
                         eval_data: List[List[float]]) -> None:
    """
    Compares performance metrics for baseline and ROSA across multiple scenarios and saves results to an Excel file, thus indicating improvements by ROSA.

    Args:
        experiment_name (str): Name of the experiment. Used to name the output Excel file and folder.
        evaluation_scenarios (List[Union[str, int]]): List of scenario identifiers used as column labels.
        metrics (List[str]): Names of the performance metrics being evaluated.
        pre_eval_data (List[List[float]]): List of metric values from the baseline without ROSA.
        eval_data (List[List[float]]): List of metric values from the evaluation of ROSA.
    """

    # Create an empty DataFrame indexed by metric names and labeled by scenario names
    df = pd.DataFrame(index=metrics, columns=evaluation_scenarios)

    # Fill the DataFrame with percentage differences (raw values for stops) for each scenario
    for metric, pre_eval_values, eval_values in zip(metrics, pre_eval_data, eval_data):
        # Validate the dimensions of input lists
        if not (len(pre_eval_values) == len(eval_values) == len(evaluation_scenarios)):
            raise ValueError(f'Mismatched lengths for {metric} or scenarios.')

        for scenario, pre_eval_value, eval_value in zip(evaluation_scenarios, pre_eval_values, eval_values):
            if metric == 'number_stops':
                # Store raw values for the number of stops
                df.at[f'{metric} (pre)', scenario] = pre_eval_value
                df.at[f'{metric} (eval)', scenario] = eval_value

            # Calculate percentage improvement, handling division by zero
            if pre_eval_value != 0:
                percentage_change = round((eval_value - pre_eval_value) / pre_eval_value * 100, 2)
            elif (pre_eval_value == 0) & (eval_value > pre_eval_value):
                percentage_change = float(100)
            else:
                percentage_change = float(0)  # Set 0 or NaN ('nan') if pre_eval_value is zero

            # Format positive values with '+' sign
            df.at[metric, scenario] = f'+{percentage_change}' if percentage_change > 0 else percentage_change

    # Compute average savings per trip (exclude raw number of stops entries)
    df['Average Savings per Trip'] = df.drop(
        index=['number_stops (eval)', 'number_stops (pre)'], errors='ignore').astype(float).mean(
        axis=1).round(2)
    # Add '+' to positive average savings values
    average_savings_values = df['Average Savings per Trip'].values
    df['Average Savings per Trip'] = [f'+{value}' if isinstance(value, (int, float)) and value > 0 else value
                                               for value in average_savings_values]

    # Ensure output directory exists
    output_dir = os.path.join('runs', experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save DataFrame to Excel file
    excel_file_path = os.path.join(output_dir, f'Performance_Evaluation_{experiment_name}.xlsx')
    with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Performance_Evaluation', index_label='Metric')


if __name__ == "__main__":
    raise NotImplementedError("This script is not intended to be run directly.")