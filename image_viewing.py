import logging
import SimpleITK as sitk
from scipy.ndimage import center_of_mass
from PIL import Image
import numpy as np


class ImageViewer:
    """
    Класс ImageViewer предназначен для отображения медицинских изображений и масок

    Атрибуты:
    - image: Медицинское изображение в форме sitk.Image
    - mask: Медицинская маска в форме sitk.Image

    Методы:
    - __init__(self, image: sitk.Image, mask: sitk.Image): Инициализация класса ImageViewer
    - show(self): Сохранение среднего среза маски и медицинского изображения в JPEG
    """
    def __init__(self, image: sitk.Image,  mask: sitk.Image):
        """
        Инициализация класса ImageViewer

        Параметры:
        - image: Медицинское изображение в форме sitk.Image
        - mask: Медицинская маска в форме sitk.Image
        """
        logging.info('ImageViewer: Loading image and mask')
        self.image = sitk.GetArrayFromImage(image)
        self.mask = sitk.GetArrayFromImage(mask)

    def show(self) -> None:
        """
        Сохранение среднего среза маски и медицинского изображения в JPEG
        """
        logging.info('ImageViewer: Finding center slice')
        mask_z = int(round(center_of_mass(self.mask)[0]))
        mri_slice = self.image[mask_z, ::-1, :]
        mask_slice = self.mask[mask_z, ::-1, :]

        logging.info('ImageViewer: Normalizing the MRI slice to 0-255 and convert to uint8')
        mri_slice_normalized = (mri_slice - np.min(mri_slice)) / (np.max(mri_slice) - np.min(mri_slice)) * 255
        mri_slice_image = Image.fromarray(mri_slice_normalized.astype(np.uint8))

        logging.info('ImageViewer: Creating a red mask')
        red_mask = np.zeros((*mask_slice.shape, 3), dtype=np.uint8)
        red_mask[mask_slice == 1] = [255, 0, 0]

        red_mask_image = Image.fromarray(red_mask)

        alpha_mask = np.zeros(mask_slice.shape, dtype=np.uint8)
        alpha_mask[mask_slice == 1] = 26  # 30% opacity (255 * 0.1 = 26)

        logging.info('ImageViewer: Creating an RGBA image for the red mask with the alpha channel')
        red_mask_image.putalpha(Image.fromarray(alpha_mask))

        combined_image = Image.new("RGBA", mri_slice_image.size)
        combined_image.paste(mri_slice_image.convert("RGBA"), (0, 0))
        combined_image.paste(red_mask_image, (0, 0), red_mask_image)

        logging.info('ImageViewer: Saving the final image as JPG')
        combined_image.convert("RGB").save("static/show_slice.jpg", "JPEG")
