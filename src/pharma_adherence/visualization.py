import matplotlib.pyplot as plt
import pandas as pd


def plot_hist(df: pd.DataFrame, column: str, label_rotation = "horizontal") -> plt:
    #TODO: Plot a histogram for a numeric column.
    data = df[column].dropna()
    plt.hist(data ,bins=30, color='skyblue', edgecolor='black')
    
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=label_rotation)
    plt.title(f'Distribution of {column}')
    return plt
"""
    Create a histogram for a selected column in the dataset.

    This function is used to visualize the distribution of values in one column.
    It first removes missing values from the selected column, then plots the
    remaining values as a histogram with 30 bins. This is useful for checking
    how frequently different values appear and whether a variable is skewed,
    spread out, or concentrated around certain values.

    Parameters:
        df: The pandas DataFrame containing the data to plot.
        column: The name of the column to visualize.
        label_rotation: Rotation angle or style for the x-axis labels.

    Returns:
        The matplotlib pyplot object containing the histogram.
    """
def plot_bar(df: pd.DataFrame, categories: str, values: float, label_rotation = "horizontal") -> plt:
    #TODO: Plot a bar plot for numeric categories and values.
    #      For multiple values per category, plot the average value per category
    new_df = df.groupby(categories, dropna=False)[values].mean().sort_values()
    plt.bar(new_df.index, new_df, color='skyblue')

    plt.xlabel(categories)
    plt.ylabel(values)
    plt.xticks(rotation=label_rotation)
    plt.title(f'Bar Chart of {values} per {categories}')
    return plt
"""
    Create a bar chart showing the average value for each category.

    This function groups the dataset by a categorical column and calculates the
    mean of a numeric value column for each group. It then creates a bar chart
    where each bar represents one category. This is useful for comparing average
    outcomes across groups, such as average proportion of days covered by drug
    name or pharmacy name.

    Parameters:
        df: The pandas DataFrame containing the data to plot.
        categories: The categorical column used to group the data.
        values: The numeric column whose average will be plotted for each group.
        label_rotation: Rotation angle or style for the x-axis labels.

    Returns:
        The matplotlib pyplot object containing the bar chart.
    """
def plot_scatter(df: pd.DataFrame, x: float, y: float) -> plt:
    #TODO: Plot a scatter plot for numeric x and y values.
    plt.scatter(df[x], df[y])

    plt.title(f'{y} vs {x}')
    plt.xlabel(x)
    plt.ylabel(y)
    return plt
"""
    Create a scatter plot comparing two numeric variables.

    This function plots one variable on the x-axis and another variable on the
    y-axis. It is useful for visually checking whether two variables may have a
    relationship, such as whether patient age or copay amount appears associated
    with proportion of days covered.

    Parameters:
        df: The pandas DataFrame containing the data to plot.
        x: The column name to use for the x-axis.
        y: The column name to use for the y-axis.

    Returns:
        The matplotlib pyplot object containing the scatter plot.
    """