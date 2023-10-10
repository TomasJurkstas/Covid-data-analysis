import pandas as pd
import numpy as np
import inspect
import sys
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import datetime


def plot_data(mapping_data: pd.DataFrame, zoom_level: int = 0.5) -> None:
    """
       Plot some data on a map of South Korea.

       Args:
           mapping_data (pd.DataFrame): DataFrame containing mapping data with
           'longitude' and 'latitude' columns.
           zoom_level (int, optional): Zoom level for the plot.
           Defaults to 0.5.

       Returns:
           None
    """
    # Load a shapefile of the map or any other geospatial data
    map_data = gpd.read_file(r'maps\ne_10m_admin_0_countries.shp')
    map_data = map_data[map_data['SOVEREIGNT'] == 'South Korea']
    # Load a shapefile of the district boundaries
    district_data = gpd.read_file(r'maps\ne_10m_admin_1_states_provinces.shp')
    district_data = district_data[district_data['iso_a2'] == 'KR']
    # Load a shapefile of the city names
    name_data = gpd.read_file(r'maps\ne_10m_populated_places.shp')
    name_data = name_data[name_data['SCALERANK'] <= 6]
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    # Plot the map data
    map_data.plot(ax=ax)
    # Plot the district boundaries
    district_data.plot(ax=ax, edgecolor='black', facecolor='none')
    # Plot the city names
    name_data.plot(ax=ax, color='black', markersize=3)
    # Set the extent and aspect ratio of the plot
    extent = [mapping_data['longitude'].min() - zoom_level,
              mapping_data['longitude'].max() + zoom_level,
              mapping_data['latitude'].min() - zoom_level,
              mapping_data['latitude'].max() + zoom_level]
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')
    # Create the geometry column using the latitude
    # and longitude columns from the DataFrame
    point_geometry = [Point(xy) for xy in zip(mapping_data['longitude'],
                                              mapping_data['latitude'])]
    # Create a GeoDataFrame from the DataFrame with the geometry column
    points = gpd.GeoDataFrame(mapping_data, geometry=point_geometry)
    # Plot the data
    points.plot(ax=ax,
                color='red',
                markersize=mapping_data.confirmed,
                alpha=0.5)
    # Add city names as text annotations
    for x, y, name in zip(name_data.geometry.x,
                          name_data.geometry.y,
                          name_data['NAME']):
        ax.annotate(text=name,
                    xy=(x, y),
                    xytext=(3, 3),
                    weight='bold',
                    textcoords='offset points',
                    fontsize=10)


def find_word(dataframe: pd.DataFrame, column: str, word: str) -> pd.DataFrame:
    """
    Filters a dataframe based on a specific word occurrence in a given column.

    Args:
        dataframe (pd.DataFrame): A pandas DataFrame to filter.
        column (str): The name of the column in the dataframe
        to search for the word.
        word (str): The word to search for in the specified column.

    Returns:
        filtered_df (pd.DataFrame): A new pandas DataFrame containing rows
        where the specified word is found in the specified column.
    """
    pattern = fr'\b{word}s?\b'
    mask = dataframe[column].str.contains(pattern, regex=True)
    filtered_df = dataframe[mask]
    return filtered_df


def count_words(dataframe: pd.DataFrame, column: str, n: int = 3) -> list:
    """
    Counts the occurrences of the last word in each string
    in a specified column of a dataframe.

    Args:
        dataframe (pd.DataFrame): A pandas DataFrame to process.
        column (str): The name of the column in the dataframe to analyze.
        n (int, optional): The number of most frequent last words to return.
        Defaults to 3.

    Returns:
        words (list): A list of the n most frequent last words
        in the specified column.
    """
    last_words = dataframe[column].str.split().str[-1]
    word_counts = last_words.value_counts()
    words = word_counts.head(n).index.tolist()
    return words


def get_date_range(chosen_date: datetime.datetime, days: int) -> list:
    """
    Calculate a date range of specified number of days before the chosen date.

    Parameters:
        chosen_date (datetime.datetime): The chosen date.
        days (int): The number of days in the date range.

    Returns:
        date_range (list): A list of datetime objects
        representing the date range.

    Example:
        chosen_date = datetime.datetime(2023, 5, 1)
        date_range = get_date_range(chosen_date, 7)
        print(date_range)
        # Output: [datetime.datetime(2023, 4, 24),
        #          datetime.datetime(2023, 4, 25),
        #          datetime.datetime(2023, 4, 26),
        #          datetime.datetime(2023, 4, 27),
        #          datetime.datetime(2023, 4, 28),
        #          datetime.datetime(2023, 4, 29),
        #          datetime.datetime(2023, 4, 30)]
    """
    start_date = chosen_date - datetime.timedelta(days=days)
    date_range = []

    for i in range(7):
        date = start_date + datetime.timedelta(days=i)
        date_range.append(date)

    return date_range


def get_matching_entries(dates_of_interest: list,
                         data: pd.DataFrame,
                         date_column: str,
                         days: int) -> pd.DataFrame:
    """
    Filter a DataFrame based on a list of dates
    and return the matching entries.

    Parameters:
        dates_of_interest (list): A list of date strings of interest.
        data (pd.DataFrame): The DataFrame to be filtered.
        date_column (str): The column name in the DataFrame containing
        the dates.
        days (int): The number of days in the date range.

    Returns:
        matching_entries (pd.DataFrame): A DataFrame containing
        the matching entries based on the specified date range.

    Example:
        dates_of_interest = ['2023-05-01', '2023-05-10', '2023-05-15']
        policy_data = pd.DataFrame(...)  # The DataFrame to be filtered
        matching_entries = get_matching_entries(dates_of_interest,
                                                policy_data,
                                                'start_date',
                                                7)
        print(matching_entries)
        # Output: DataFrame containing the matching entries
        based on the 'start_date' column and the date range.

    Note:
        - The date strings in 'dates_of_interest' should be in the format
        'YYYY-MM-DD'.
        - The 'data' DataFrame should contain a column specified by
        'date_column' that contains the dates to be compared.
        - The 'days' parameter specifies the number of days in the date range.
    """
    matching_entries = pd.DataFrame()

    for date_str in dates_of_interest:
        date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        date_range = get_date_range(date_obj, days)
        matching_df = data[data[date_column].isin(date_range)]
        matching_entries = pd.concat([matching_entries, matching_df],
                                     ignore_index=True)

    return matching_entries


def set_parameters(title: str, xlabel: str,
                   ylabel: str, rotation: int = 45) -> None:
    """
    Set parameters for a matplotlib plot.

    Parameters:
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        rotation (int, optional): The rotation angle for the x-axis tick labels.
        Defaults to 45.

    Returns:
        None
    """
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.legend()


def calculate_differences(data: pd.DataFrame, index_column: str,
                          value_column: str) -> tuple:
    """
        Calculate differences and fatalities for specified values in
        a DataFrame.

        Parameters:
            data (pd.DataFrame): The input DataFrame containing the data.
            index_column (str): The column used for grouping and obtaining
            unique values.
            value_column (str): The column for which differences are calculated.

        Returns:
            tuple: A tuple containing the dictionaries of differences and
            fatalities, the dates for the last subset, and the unique values.

    """
    unique_values = data.index.get_level_values(index_column).unique()

    differences = {}
    differences_fatalities = {}

    for value in unique_values:
        subset = data[data.index.get_level_values(index_column) == value]
        differences[value] = np.diff(subset[value_column])
        differences_fatalities[value] = np.diff(subset['fatalities, %'])

    dates = subset.index.get_level_values('date')[1:]

    return differences, differences_fatalities, dates, unique_values


def plot_differences(unique_values: pd.Index,
                     xaxis: pd.Index, yaxis: dict) -> None:
    """
    Plot differences based on unique values.

    Parameters:
        unique_values (pd.Index): The unique values for grouping.
        xaxis (pd.Index): The x-axis values.
        yaxis (dict): The dictionary of y-axis values with unique values as
        keys.

    Returns:
        None

    """
    for value in unique_values:
        plt.plot(xaxis, yaxis[value], label=value)


imported_functions = [name for name, _ in inspect.getmembers(
    sys.modules[__name__], inspect.isfunction)]

print(f'Imported functions: {imported_functions}')
