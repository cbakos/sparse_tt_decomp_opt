from itertools import product
from typing import Any, List

import pandas as pd
import numpy as np
import plotly.express as px


def add_tt_mals_runtime_cols(df: pd.DataFrame) -> pd.DataFrame:
    df["log_obj_func"] = np.log(df["max_mode_size"]**6 +
                                df["max_mode_size"]**3 * df["rank"] +
                                df["max_mode_size"]**2 * df["rank"]**2)
    df["obj_func"] = np.exp(df["log_obj_func"])
    return df


def line_plot_padding_tile_size_tt_mals_runtime_per_matrix(df: pd.DataFrame, matrix_str: str, padding_num: int = 11):
    # get separate color for each padding level
    default_colorscale = px.colors.sequential.Jet
    colors = px.colors.sample_colorscale(default_colorscale, padding_num)

    fig = px.line(df[df["matrix_name"] == matrix_str], x="tile_size", y="log_obj_func", color="padding",
                  symbol="padding", log_x=True, color_discrete_sequence=colors,
                  labels={
                      "tile_size": "Tile size",
                  })
    fig.update_layout(
        title={
            'text': "Influence of tile size choice and padding on TT-MALS runtime ({})".format(matrix_str),
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        plot_bgcolor='white',  # Plot area background color
        paper_bgcolor='white',  # Entire figure background color
        font=dict(color='black'),  # Font color
        yaxis_title=r'$\log(I^6 + rI^3 + r^2I^2)$'
    )
    fig.show()
    fig.write_image("plots/{}_padding_tile_size_vs_log_obj_func.pdf".format(matrix_str))


def get_percentage_change_per_double_category(data_frame: pd.DataFrame, result_column: str, variable: str,
                                              baseline_col: str, baseline_value: Any, category1: str, category2: str) \
        -> pd.DataFrame:
    """

    :param data_frame: complete df
    :param result_column: column name to store results
    :param variable: name of column to check for change
    :param baseline_col: column name which determines baseline
    :param baseline_value: value in baseline column to serve as baseline
    :param category1: outer category to group by
    :param category2: inner category to group by, passed on to get_percentage_change_per_category
    :return: updated df
    """
    data_frame[result_column] = np.nan

    for category_value in data_frame[category1].unique():
        df_per_category = data_frame[data_frame[category1] == category_value]
        result_df = get_percentage_change_per_category(data_frame=df_per_category, result_column=result_column,
                                                       variable=variable, baseline_col=baseline_col,
                                                       baseline_value=baseline_value, category=category2)
        data_frame.loc[data_frame[category1] == category_value, result_column] = result_df[result_column].values
    return data_frame


def get_percentage_change_per_category(data_frame: pd.DataFrame, result_column: str, variable: str,
                                       baseline_col: str, baseline_value: Any, category: str) -> pd.DataFrame:
    """

    :param data_frame: complete df
    :param result_column: column name to store results
    :param variable: name of column to check for change
    :param baseline_col: column name which determines baseline
    :param baseline_value: value in baseline column to serve as baseline
    :param category: df column name that we need to do a grouping over
    :return: updated df
    """
    data_frame[result_column] = np.nan

    for category_value in data_frame[category].unique():
        df_per_category = data_frame[data_frame[category] == category_value]
        original_values = df_per_category[
            df_per_category[baseline_col] == baseline_value][variable].unique()

        # Ensure there is exactly one baseline value
        if len(original_values) == 1:
            original_value = original_values[0]
            data_frame.loc[data_frame[category] == category_value, result_column] = (
                    df_per_category[variable] / original_value
            ).values
        else:
            # Handle case where there are no original_values values
            raise AssertionError("there should be only one unique, baseline value per matrix")
    return data_frame
