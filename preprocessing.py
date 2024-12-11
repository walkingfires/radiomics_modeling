import logging
import numpy as np
import SimpleITK as sitk
import os
from datetime import datetime
from intensity_normalization.cli.zscore import ZScoreNormalize
from intensity_normalization.typing import Modality


class Preprocessor:
    """
    Класс Preprocessor предназначен для предобработки медицинских изображений и масок

    Атрибуты:
    - mri_modality (str): Модальность МРТ, может быть 'T1' или 'T2'
    - save (bool): Флаг, указывающий, нужно ли сохранять обработанные данные

    Методы:
    - __init__(self, mri_modality: str, save=False): Инициализация класса Preprocessor
    - normalize(self, image: sitk.Image) -> sitk.Image: Нормализация изображения
    - intensity_normalize(self, image: sitk.Image) -> sitk.Image: Нормализация интенсивности изображения
    - new_image_preprocessing(self, input_path, normalize=True, resample=True): Предобработка нового изображения
    - mask_preprocessing(self, input_path, resample=True): Предобработка маски
    - preprocessing_step(self, image_path, mask_path, normalize=True, resample=True): Предобработка изображения и маски
    """

    def __init__(self, mri_modality: str, save=False):
        """
        Инициализация класса Preprocessor

        Параметры:
        - mri_modality (str): Модальность МРТ, может быть 'T1' или 'T2'
        - save (bool): Флаг, указывающий, нужно ли сохранять обработанные данные
        """
        logging.info('Preprocessor: Initializing Preprocessor class.')
        self.save = save
        self.normalizer = ZScoreNormalize()
        if mri_modality == 'T1':
            self.modality = Modality.T1
        elif mri_modality == 'T2':
            self.modality = Modality.T2
        else:
            logging.error('Preprocessor: Invalid modality choice')
            raise ValueError("Invalid modality choice. Choose 'T1' or 'T2'")

    @staticmethod
    def normalize(image: sitk.Image) -> sitk.Image:
        """
        Z-score нормализация изображения

        Параметры:
        - image (sitk.Image): Входное изображение

        Возвращает:
        - normalized_image (sitk.Image): Нормализованное изображение
        """
        logging.info('Preprocessor: Normalizing image')
        flatten_image_array = sitk.GetArrayFromImage(image).flatten()
        mu, std = np.mean(flatten_image_array), np.std(flatten_image_array)
        normalized_image = (image - mu) / std
        # normalized_image = sitk.Normalize(image)
        return normalized_image

    def intensity_normalize(self, image: sitk.Image) -> sitk.Image:
        """
        Нормализация интенсивности изображения пакетом intensity_normalization

        Параметры:
        - image (sitk.Image): Входное изображение

        Возвращает:
        - normalized_image (sitk.Image): Нормализованное изображение
        """
        logging.info('Preprocessor: Normalizing image intensity')
        array = np.asanyarray(sitk.GetArrayFromImage(image))
        normalized_array = np.asanyarray(self.normalizer(array, modality=self.modality))
        normalized_image = sitk.GetImageFromArray(normalized_array)
        normalized_image.CopyInformation(image)
        return normalized_image

    def image_preprocessing(self, input_path: str, normalize=True, resample=True) -> sitk.Image:
        """
        Предобработка изображения

        Параметры:
        - input_path (str): Путь к входному изображению
        - normalize (bool): Флаг, указывающий, нужно ли нормализовать изображение
        - resample (bool): Флаг, указывающий, нужно ли ресэмплировать изображение

        Возвращает:
        - final_image (sitk.Image): Предобработанное изображение
        """
        logging.info('Preprocessor: Loading image')
        if isinstance(input_path, str) and os.path.isfile(input_path):
            logging.info('Preprocessor: Preprocessing image')
            image = sitk.ReadImage(input_path)

            final_image = image

            if normalize:
                logging.info('Preprocessor: Normalizing image')
                # final_image = self.normalize(image)
                final_image = self.intensity_normalize(image)

            if resample:
                logging.info('Preprocessor: Resampling image')
                resampled_image = sitk.Resample(
                    image1=final_image,
                    size=np.rint(
                        final_image.GetSize() * np.array(final_image.GetSpacing()) / (1, 1, 1)).astype(
                        int).tolist(),
                    transform=sitk.Transform(),
                    interpolator=sitk.sitkBSpline,
                    outputOrigin=final_image.GetOrigin(),
                    outputSpacing=(1, 1, 1),
                    outputDirection=final_image.GetDirection(),
                    outputPixelType=final_image.GetPixelID(),
                )
                final_image = resampled_image

            # TODO: make file path
            if self.save:
                logging.info('Preprocessor: Saving image')
                output_path = 'image' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.nii'
                sitk.WriteImage(final_image, output_path)
        else:
            logging.error('Preprocessor: Error reading image Filepath or SimpleITK object')
            raise ValueError('Preprocessor: Error reading image Filepath or SimpleITK object')

        return final_image

    def mask_preprocessing(self, input_path: str, resample=True) -> sitk.Image:
        """
        Предобработка маски

        Параметры:
        - input_path (str): Путь к входной маске
        - resample (bool): Флаг, указывающий, нужно ли ресэмплировать маску

        Возвращает:
        - result_mask (sitk.Image): Предобработанная маска
        """
        logging.info('Preprocessor: Loading mask')
        if isinstance(input_path, str) and os.path.isfile(input_path):
            logging.info('Preprocessor: Preprocessing mask')

            mask = sitk.ReadImage(input_path)

            result_mask = mask

            if resample:
                logging.info('Preprocessor: Resampling mask')
                preprocessing_mask = sitk.Resample(
                    image1=mask,
                    size=np.rint(mask.GetSize() * np.array(mask.GetSpacing()) / (1, 1, 1)).astype(int).tolist(),
                    transform=sitk.Transform(),
                    interpolator=sitk.sitkLinear,
                    outputOrigin=mask.GetOrigin(),
                    outputSpacing=(1, 1, 1),
                    outputDirection=mask.GetDirection(),
                    outputPixelType=mask.GetPixelID(),
                )
                result_mask = preprocessing_mask

            # TODO: make file path and check labels in mask
            if self.save:
                logging.info('Preprocessor: Saving mask')
                output_path = 'mask' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.nii'
                sitk.WriteImage(result_mask, output_path)
        else:
            logging.error('Preprocessor: Error reading mask Filepath or SimpleITK object')
            raise ValueError('Preprocessor: Error reading mask Filepath or SimpleITK object')

        return result_mask

    def preprocessing_step(self, image_path: str, mask_path: str, normalize=True, resample=True) -> tuple:
        """
        Предобработка изображения и маски

        Параметры:
        - image_path (str): Путь к входному изображению
        - mask_path (str): Путь к входной маске
        - normalize (bool): Флаг, указывающий, нужно ли нормализовать изображение
        - resample (bool): Флаг, указывающий, нужно ли ресэмплировать изображение и маску

        Возвращает:
        - new_image (sitk.Image): Предобработанное изображение
        - new_mask (sitk.Image): Предобработанная маска
        """
        new_image = self.image_preprocessing(image_path, normalize=normalize, resample=resample)
        new_mask = self.mask_preprocessing(mask_path, resample=resample)
        return new_image, new_mask
