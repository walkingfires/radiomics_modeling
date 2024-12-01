import logging
import json
from radiomics import featureextractor


class FeatureExtractor:

    def __init__(self, model, desired_order_bool=False):
        logging.info('Initializing FeatureExtractor class.')
        self.model = model
        self.desired_order_bool = desired_order_bool
        if isinstance(model, str):
            logging.info('Initializing extractor with params.')
            try:
                self.params = f'params/{model}_extracting_params.yaml'
                self.extractor = featureextractor.RadiomicsFeatureExtractor(self.params)
                self.features = self.extractor.enabledFeatures
                if self.desired_order_bool:
                    logging.info('Initializing desired_order.')
                    self.desired_order = json.load(open(f'params/{model}_features.json'))
                else:
                    self.desired_order = None
            except Exception:
                raise ValueError('Extractor: Model not found.')
        else:
            raise ValueError('Extractor: Model not initialized.')

    def extract_features(self, image, mask):
        extracted_features = self.extractor.execute(image, mask)

        if self.desired_order:
            model_features = {key: extracted_features[key] for key in self.desired_order if key in extracted_features}
        else:
            model_features = {key: value for key, value in extracted_features.items() if 'diagnostics' not in key}

        return model_features
