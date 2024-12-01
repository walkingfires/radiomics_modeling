import os

import joblib
import pandas as pd
class Predictor:

    def __init__(self, model: str):
        if isinstance(model, str) and os.path.isfile(f'models/{model}_model.joblib'):
            self.classifier = joblib.load(f'models/{model}_model.joblib')
        else:
            raise ValueError('Classifier: Model not found.')

    def predict(self, data: dict, **kwargs):
        prediction = self.classifier.predict(pd.DataFrame([data]))
        return prediction[0]
