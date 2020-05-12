from data_crawler import load_data, DATA_DIR
from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm
from task_geo.dataset_builders.nasa import nasa
from itertools import chain

NASA_WEATHER_API_BASE_URL = "https://power.larc.nasa.gov/cgi-bin/v1/DataAccess.py"
identifier = "identifier=SinglePoint"
user_community = "userCommunity=SSE"
temporal_average = "tempAverage=DAILY"
output_format = "outputList=JSON,ASCII"
user = "user=anonymous"

PARAMETERS = {
    "temperature": ["T2M", "T2M_MIN", "T2M_MAX"],
    "humidity": ["RH2M", "QV2M"],
    "pressure": ["PS"]
}

query_fields = list(chain.from_iterable([PARAMETERS[p] for p in PARAMETERS.keys()]))
params_str = f"parameters={','.join(query_fields)}"


def get_weather_time_series_for_one_location(df):
    """

    Args:
        df: pandas.DataFrame with columns  ['country', 'region', 'sub_region', 'lon', 'lat'] (to fit
        with task_geo library)-- here, supply df for 1 unique lon/lat only to make it more efficient to parallelize

    Returns:

    """
    weather = nasa(df, start_date=df.date.min(), end_date=df.date.max(), join=False)
    if weather is None:
        return
    weather = weather[[
        'date', 'avg_temperature', 'min_temperature',
        'max_temperature', 'relative_humidity', 'specific_humidity',
        'pressure']]
    df = df.set_index('date').merge(weather.set_index('date'), left_index=True, right_index=True)
    return df.reset_index()


def get_weather_time_series_all(df, parallel_jobs=20):
    unqiue_lat_long = df[['lat', 'long']].drop_duplicates().sample(frac=1.0)
    unique_dataframes = [df[(df.lat == lat) & (df.long==long)].rename(columns={'long': 'lon'})  for idx,( lat, long) in unqiue_lat_long.iterrows()]
    weather_results = Parallel(n_jobs=parallel_jobs, prefer='threads')(delayed(get_weather_time_series_for_one_location)(df) for df in tqdm(unique_dataframes))
    return pd.concat([x for x in weather_results if x is not None])

if __name__=="__main__":

    df = load_data()
    weather_df = get_weather_time_series_all(df)
    weather_df.to_csv(DATA_DIR / 'timeseries_with_weather_mobility.csv', index=False)
