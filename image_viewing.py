import SimpleITK as sitk
from scipy.ndimage import center_of_mass
from PIL import Image
import numpy as np


class ImageViewer:
    def __init__(self, image: sitk.Image,  mask: sitk.Image):
        self.image = sitk.GetArrayFromImage(image)
        self.mask = sitk.GetArrayFromImage(mask)

    def show(self):
        mask_z = int(round(center_of_mass(self.mask)[0]))
        mri_slice = self.image[mask_z, ::-1, :]
        mask_slice = self.mask[mask_z, ::-1, :]

        # Normalize the MRI slice to 0-255 and convert to uint8
        mri_slice_normalized = (mri_slice - np.min(mri_slice)) / (np.max(mri_slice) - np.min(mri_slice)) * 255
        mri_slice_image = Image.fromarray(mri_slice_normalized.astype(np.uint8))

        # Create a red mask
        red_mask = np.zeros((*mask_slice.shape, 3), dtype=np.uint8)  # Create an empty RGB image
        red_mask[mask_slice == 1] = [255, 0, 0]  # Set red color where mask is 1

        # Convert the red mask to an image
        red_mask_image = Image.fromarray(red_mask)

        # Create an alpha channel for the red mask with 30% transparency
        alpha_mask = np.zeros(mask_slice.shape, dtype=np.uint8)
        alpha_mask[mask_slice == 1] = 26  # 30% opacity (255 * 0.1 = 26)

        # Create an RGBA image for the red mask with the alpha channel
        red_mask_image.putalpha(Image.fromarray(alpha_mask))

        # Combine the images
        combined_image = Image.new("RGBA", mri_slice_image.size)
        combined_image.paste(mri_slice_image.convert("RGBA"), (0, 0))
        combined_image.paste(red_mask_image, (0, 0), red_mask_image)

        # Save the final image as JPG
        combined_image.convert("RGB").save("static/show_slice.jpg", "JPEG")
