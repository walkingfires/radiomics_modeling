import logging
import os
import joblib
import pandas as pd

class Predictor:
    """
    Класс Predictor предназначен для предсказания на основе загруженной модели.

    Атрибуты:
    - classifier: Загруженная модель

    Методы:
    - __init__(self, model: str): Инициализация класса Predictor
    - predict(self, data: dict, **kwargs): Предсказание на основе входных данных
    """

    def __init__(self, model: str):
        """
        Инициализация класса Predictor

        Параметры:
        - model: Название модели, используемой для предсказания
        """
        logging.info("Predictor: Loading model")
        if isinstance(model, str) and os.path.isfile(f'models/{model}_model.joblib'):
            self.classifier = joblib.load(f'models/{model}_model.joblib')
        else:
            logging.error('Predictor: Model not found')
            raise ValueError('Predictor: Model not found.')

    def predict(self, data: dict):
        """
        Предсказание на основе входных данных

        Параметры:
        - data: Входные данные для предсказания

        Возвращает:
        - prediction: Предсказание
        """
        logging.info("Predictor: Predicting")
        prediction = self.classifier.predict(pd.DataFrame([data]))
        return prediction[0]
