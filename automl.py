from pycaret.regression import *
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_log_error

class MLSystem:
    def __init__(self):
        pass

    def load_data(self, data_path):
        
        try:
            return pd.read_csv(data_path)
        except Exception as e:
            raise IOError(f"Error def-data: {str(e)}")

    def preprocess_data(self, data):
     try:
        # Codificar la variable categórica 'Sex' como variables binarias
        data['Sex'] = data['Sex'].map({'M': 1, 'F': 0})

        # Corregir los nombres de las columnas de peso
        data.rename(columns={'Whole weight.1': 'Shucked weight', 'Whole weight.2': 'Viscera weight'}, inplace=True)

        reg = setup(
            data=data,
            target="Rings",
            numeric_features=["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight"],
            normalize=True,
            normalize_method="zscore",
            polynomial_features=True
        )

        return reg

     except Exception as e:
        raise RuntimeError(f"Error def-preprocesses: {str(e)}")
   
   
    def tuned_model(self, data):
     try:
        # Crear el modelo LightGBM 
        model = create_model('lightgbm')
        
        # Definir una cuadrícula de hiperparámetros para el ajuste
        param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7]
        }

        # Ajustar el modelo con la cuadrícula de hiperparámetros
        tuned_model = tuned_model(model, custom_grid=param_grid, optimize='RMSLE')

        return tuned_model

     except Exception as e:
        raise RuntimeError(f"Error def-tuned: {str(e)}")

    def evaluate_model(self, model, test_data_path):
      try:
        # Leer datos de prueba
        test_data = pd.read_csv(test_data_path)
        # Codificar la variable categórica 'Sex' como variables binarias
        test_data['Sex'] = test_data['Sex'].map({'M': 1, 'F': 0})

        # Corregir los nombres de las columnas de peso
        test_data.rename(columns={'Whole weight.1': 'Shucked weight', 'Whole weight.2': 'Viscera weight'}, inplace=True)
        
        # Realizar predicciones con el modelo
        predictions = predict_model(model, data=test_data)
        
        # Renombrar la columna de predicciones como "Rings"
        predictions.rename(columns={'Label': 'Rings'}, inplace=True)
        
        # Guardar las predicciones en un archivo CSV
        submission_path = 'submission.csv'
        predictions[['id', 'Rings']].to_csv(submission_path, index=False)
        
        # Calcular la métrica RMSLE
        rmsle = np.sqrt(mean_squared_log_error(test_data["Rings"], predictions["Rings"]))
        
        # Retornar tanto las predicciones como el RMSLE
        return predictions, rmsle
      except Exception as e:
        raise RuntimeError(f"Error def-eval: {str(e)}")

    def run_entire_workflow(self, train_data_path, test_data_path):
        try:
            train_data = self.load_data(train_data_path)
            model_inicial = self.preprocess_data(train_data)
            model_final = self.tuned_model(model_inicial, 'RMSLE')
            rmsle = self.evaluate_model(model_final, test_data_path)
            return {'RMSLE': rmsle}
        except Exception as e:
            return {'message': str(e)}

    
        