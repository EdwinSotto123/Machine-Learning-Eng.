import airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import dotenv
from sklearn.metrics import accuracy_score
from joblib import dump, load
import zipfile
import os
from automl import MLSystem
from kaggle.api.kaggle_api_extended import KaggleApi

os.environ['KAGGLE_USERNAME'] = 'edwinisaacsotocossio'
os.environ['KAGGLE_KEY'] = '0e5ccd1f05dcbaebaa0e3c66ca72b5a1'

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 4, 26),
    'email':['edwin.soto.c@uni.pe'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_workflow',
    default_args=default_args,
    description='A simple ML pipeline',
    schedule_interval='0 23 * * *',
)

def  GetDataKaggle():    
    api = KaggleApi()
    api.authenticate()
    competition_name = 'playground-series-s4e4'
    download_path = 'data/'
    api.competition_download_files(competition_name, path=download_path)
    for item in os.listdir(download_path):
        if item.endswith('.zip'):
            zip_ref = zipfile.ZipFile(os.path.join(download_path, item), 'r')
            zip_ref.extractall(download_path)
            zip_ref.close()
            print(f"Unzipped {item}")
    return api

def  AutoML_PyCaret():
    mySystem = MLSystem()
    mySystem.run_entire_workflow("data/train.csv", "data/test.csv")
    return None

def submit_kaggle(api, submission_file ="submission_0.csv", message="First submission", competition="playground-series-s4e4"):
    
    try:
        api.competition_submit(file_name=submission_file, message=message, competition=competition)
    except Exception as e:
        raise RuntimeError(f"Error def-submit: {str(e)}")

GetDataKaggle = PythonOperator(
    task_id='GetDataKaggle',
    python_callable=GetDataKaggle,
    dag=dag,
)

AutoML_PyCaret = PythonOperator(
    task_id='AutoML_PyCaret',
    python_callable=AutoML_PyCaret,
    dag=dag,
)

SubmitKaggle = PythonOperator(
    task_id='SubmitKaggle',
    python_callable=submit_kaggle,
    dag=dag,
)

GetDataKaggle >> AutoML_PyCaret >> SubmitKaggle
