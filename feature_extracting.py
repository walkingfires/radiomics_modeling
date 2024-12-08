import logging
import json
import SimpleITK as sitk
from radiomics import featureextractor


class FeatureExtractor:
    """
    Класс FeatureExtractor предназначен для извлечения признаков из медицинских изображений и масок

    Атрибуты:
    - model: Название модели, используемой для извлечения признаков
    - desired_order_bool: Флаг, указывающий, нужно ли использовать желаемый порядок признаков

    Методы:
    - __init__(self, model, desired_order_bool=False): Инициализация класса FeatureExtractor
    - extract_features(self, image, mask): Извлечение признаков из изображения и маски
    """

    def __init__(self, model, desired_order_bool=False):
        """
        Инициализация класса FeatureExtractor

        Параметры:
        - model: Модель, используемая для извлечения признаков
        - desired_order_bool: Флаг, указывающий, нужно ли использовать желаемый порядок признаков
        """
        logging.info('Initializing FeatureExtractor class')
        self.model = model
        self.desired_order_bool = desired_order_bool
        if isinstance(model, str):
            logging.info('Extractor: Initializing extractor with params')
            try:
                self.params = f'params/{model}_extracting_params.yaml'
                self.extractor = featureextractor.RadiomicsFeatureExtractor(self.params)
                self.features = self.extractor.enabledFeatures
                if self.desired_order_bool:
                    logging.info('Extractor: Initializing desired_order')
                    self.desired_order = json.load(open(f'params/{model}_features.json'))
                else:
                    self.desired_order = None
            except Exception:
                logging.error('Extractor: Model not found')
                raise ValueError('Extractor: Model not found')
        else:
            logging.error('Extractor: Model not initialized')
            raise ValueError('Extractor: Model not initialized')

    def extract_features(self, image: sitk.Image, mask: sitk.Image) -> dict:
        """
        Извлечение признаков из изображения и маски

        Параметры:
        - image: Входное изображение
        - mask: Входная маска

        Возвращает:
        - model_features: Словарь с извлеченными признаками
        """
        logging.info('Extractor: Extracting features')
        extracted_features = self.extractor.execute(image, mask)

        if self.desired_order:
            logging.info('Extractor: Saving extracted features in desired_order in dictionary')
            model_features = {key: extracted_features[key] for key in self.desired_order if key in extracted_features}
        else:
            logging.info('Extractor: Saving extracted features in dictionary')
            model_features = {key: value for key, value in extracted_features.items() if 'diagnostics' not in key}

        return model_features
