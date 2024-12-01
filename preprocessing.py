import logging
import numpy as np
import SimpleITK as sitk
import os
from datetime import datetime
from intensity_normalization.cli.zscore import ZScoreNormalize
from intensity_normalization.typing import Modality


class Preprocessor:
    """
    Инициализация класса Preprocessor для предобработки изображений и масок.
    """

    def __init__(self, mri_modality: str, save=False):
        logging.info('Initializing Preprocessor class.')
        self.save = save
        self.normalizer = ZScoreNormalize()
        if mri_modality == 'T1':
            self.modality = Modality.T1
        elif mri_modality == 'T2':
            self.modality = Modality.T2
        else:
            raise ValueError("Invalid modality choice. Choose 'T1' or 'T2'.")

    @staticmethod
    def normalize(image: sitk.Image) -> sitk.Image:
        """
        Normalize the given SimpleITK image.

        This method flattens the input image into a 1D array, computes the mean and standard deviation,
        and then normalizes the image by subtracting the mean and dividing by the standard deviation.

        Parameters:
        ----------
        image : sitk.Image
            The input image to be normalized, represented as a SimpleITK image object.

        Returns:
        -------
        sitk.Image
            The normalized image, represented as a SimpleITK image object.
        """
        flatten_image_array = sitk.GetArrayFromImage(image).flatten()
        mu, std = np.mean(flatten_image_array), np.std(flatten_image_array)
        normalized_image = (image - mu) / std
        # normalized_image = sitk.Normalize(image)
        return normalized_image

    def intensity_normalize(self, image: sitk.Image) -> sitk.Image:
        array = np.asanyarray(sitk.GetArrayFromImage(image))
        normalized_array = np.asanyarray(self.normalizer(array, modality=self.modality))
        normalized_image = sitk.GetImageFromArray(normalized_array)
        normalized_image.CopyInformation(image)
        return normalized_image

    def new_image_preprocessing(self, input_path, normalize=True, resample=True):
        """
        Предобработка медицинского изображения: загрузка, ресэмплинг, нормализация и, при необходимости, сохранение
        предобработанного изображения.

        :param input_path: Путь к входному изображению (строка).
        :return: Предобработанное изображение в виде объекта SimpleITK.
        :raises ValueError: Если input_path не является допустимым путем к файлу или изображение не может быть прочитано.
        """
        logging.info('Loading image')
        if isinstance(input_path, str) and os.path.isfile(input_path):
            logging.info('Preprocessing image')
            image = sitk.ReadImage(input_path)

            final_image = image

            if normalize:
                logging.info('Normalizing image')
                # final_image = self.normalize(image)
                final_image = self.intensity_normalize(image)

            if resample:
                logging.info('Resampling image')
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
                output_path = 'image' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.nii'
                sitk.WriteImage(final_image, output_path)
        else:
            raise ValueError('Error reading image Filepath or SimpleITK object')

        return final_image

    def mask_preprocessing(self, input_path, resample=True):
        """
        Предобработка маски: загрузка, ресэмплинг и, при необходимости, сохранение предобработанной маски.

        :param input_path: Путь к входной маске (строка).
        :return: Предобработанная маска в виде объекта SimpleITK.
        :raises ValueError: Если input_path не является допустимым путем к файлу или маска не может быть прочитана.
        """
        logging.info('Loading mask')
        if isinstance(input_path, str) and os.path.isfile(input_path):
            logging.info('Preprocessing mask')

            mask = sitk.ReadImage(input_path)  # Загрузка маски

            result_mask = mask

            if resample:
                logging.info('Resampling mask')
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
                output_path = 'mask' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.nii'
                sitk.WriteImage(result_mask, output_path)
        else:
            raise ValueError('Error reading mask Filepath or SimpleITK object')

        return result_mask

    def preprocessing_step(self, image_path, mask_path, normalize=True, resample=True):
        new_image = self.new_image_preprocessing(image_path, normalize=normalize, resample=resample)
        new_mask = self.mask_preprocessing(mask_path, resample=resample)
        return new_image, new_mask
