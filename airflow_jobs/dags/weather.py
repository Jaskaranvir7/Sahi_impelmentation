from airflow import DAG
from airflow.providers.http.hooks.http import HttpHook
from airflow.decorators import task
from airflow.utils.dates import days_ago
import requests
import json

# Latitude and longitude for the desired location (London in this case)
LATITUDE = '-37.810272'
LONGITUDE = '144.962646'
API_CONN_ID='open_meteo_api'

default_args={
    'owner':'airflow',
    'start_date':days_ago(1)
}

## DAG
with DAG(dag_id='weather_etl_pipeline',
         default_args=default_args,
         schedule_interval='@daily',
         catchup=False) as dags:
    
    @task()
    def extract_weather_data():
        """Extract weather data from Open-Meteo API using Airflow Connection."""

        # Use HTTP Hook to get connection details from Airflow connection

        http_hook=HttpHook(http_conn_id=API_CONN_ID,method='GET')

        ## Build the API endpoint
        endpoint=f'/v1/forecast?latitude={LATITUDE}&longitude={LONGITUDE}&current_weather=true'

        ## Make the request via the HTTP Hook
        response=http_hook.run(endpoint)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to fetch weather data: {response.status_code}")
        
    @task()
    def transform_weather_data(weather_data):
        """Transform the extracted weather data."""
        current_weather = weather_data['current_weather']
        transformed_data = {
            'latitude': LATITUDE,
            'longitude': LONGITUDE,
            'temperature': current_weather['temperature'],
            'windspeed': current_weather['windspeed'],
            'winddirection': current_weather['winddirection'],
            'weathercode': current_weather['weathercode']
        }
        return transformed_data
    
    @task()
    def print_weather_data(transformed_data):
        """Print transformed data."""
        print("Transformed Weather Data:")
        print(json.dumps(transformed_data, indent=4))

    ## DAG Workflow - ETL Pipeline
    weather_data = extract_weather_data()
    transformed_data = transform_weather_data(weather_data)
    print_weather_data(transformed_data)
