from itertools import product
from typing import Any, List, Dict

import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px
import plotly.graph_objects as go
from time_complexity_utils import *


def add_tt_mals_runtime_cols(df: pd.DataFrame) -> pd.DataFrame:
    df["log_obj_func"] = np.log(df["max_mode_size"] ** 6 +
                                df["max_mode_size"] ** 3 * df["rank"] +
                                df["max_mode_size"] ** 2 * df["rank"] ** 2)
    df["obj_func"] = np.exp(df["log_obj_func"])

    # also add full runtimes with O(log(n)) assumption for s and c
    df["full_runtime"] = tt_solve_with_conversions_time_complexity(s=np.log(df["n"]),
                                                                   r=df["rank"],
                                                                   I=df["max_mode_size"],
                                                                   d=np.log(df["n"]),
                                                                   num_it=np.log(df["n"]),
                                                                   z=df["z_reduced"],
                                                                   n=df["n"])
    df["log_full_runtime"] = np.log(df["full_runtime"])
    return df


def line_plot_padding_tile_size_tt_mals_runtime_per_matrix(df: pd.DataFrame, matrix_str: str, padding_num: int = 11):
    # get separate color for each padding level
    default_colorscale = px.colors.sequential.Jet
    colors = px.colors.sample_colorscale(default_colorscale, padding_num)

    df.sort_values(by=["matrix_name", "padding", "tile_size"], inplace=True)

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


def line_plot_tile_size_rank_percentage_per_matrix(data_frame: pd.DataFrame, column_str: str, matrix_color_map: Dict) \
        -> None:
    fig = px.line(data_frame, x="tile_size", y="rank_percentage", color="matrix_name", symbol="matrix_name", log_x=True,
                  color_discrete_map=matrix_color_map,
                  labels={
                      "rank_percentage": "Rank (r) ratio",
                      "matrix_name": "Matrix name",
                      "tile_size": "Tile size",
                  })
    fig.update_layout(
        title={
            'text': "Influence of {} on ranks for different tile sizes".format(column_str.upper()),
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        plot_bgcolor='white',  # Plot area background color
        paper_bgcolor='white',  # Entire figure background color
        font=dict(color='black'),  # Font color
    )
    fig.show()
    fig.write_image("plots/{}_tile_size_rank_ratio.pdf".format(column_str))


def normality_check_histogram(data_frame: pd.DataFrame, column_str: str, title_str: str, nbins: int = 40) -> None:
    fig = px.histogram(data_frame, x=column_str, histnorm="probability", nbins=nbins,
                       labels={
                           "log_obj_func": r'$\log(I^6 + rI^3 + r^2I^2)$',
                       }
    )

    mean = data_frame[column_str].mean()
    std_dev = data_frame[column_str].std()

    # Create a normal distribution curve
    x = np.linspace(min(data_frame[column_str]), max(data_frame[column_str]), 100)
    y = norm.pdf(x, mean, std_dev)

    # Add the normal distribution curve to the histogram
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Normal Distribution'))

    fig.update_layout(
        title={
            'text': title_str,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        plot_bgcolor='white',  # Plot area background color
        paper_bgcolor='white',  # Entire figure background color
        font=dict(color='black'),  # Font color
    )
    fig.show()
    fig.write_image("plots/{}_{}.pdf".format(title_str.lower().replace(" ", "_").replace("<br>", ""), column_str))
