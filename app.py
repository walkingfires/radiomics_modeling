import os
import streamlit as st

from feature_extracting import FeatureExtractor
from preprocessing import Preprocessor
from predicting import Predictor
from image_viewing import ImageViewer


# Function to process the uploaded files
def process_files(image_name: str, mask_name: str, clinical_data: dict):
    model = 'liver_t2w_xgboost'
    preprocessor = Preprocessor(mri_modality='T2')
    feature_extractor = FeatureExtractor(model, desired_order_bool=False)
    predictor = Predictor(model)

    image_path = os.path.join("static", image_name)
    mask_path = os.path.join("static", mask_name)

    image, mask = preprocessor.preprocessing_step(image_path, mask_path, normalize=True, resample=False)

    visualizer = ImageViewer(image, mask)

    # Save the visualized image
    visualizer.show()

    # Remove the uploaded files after processing
    os.remove(image_path)
    os.remove(mask_path)

    features = feature_extractor.extract_features(image, mask)
    data = features | clinical_data

    return predictor.predict(data)


def save_uploaded_file(uploaded_file, save_path):
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


# Main function to run the Streamlit app
def main():
    st.set_page_config(
        page_title="Radiomics Predictor",
        page_icon="🧊")

    st.markdown(
        """
        <h1 style="text-align: center; margin-top: 0;">
            Radiomics Predictor
        </h1>
        """,
        unsafe_allow_html=True
    )

    _, _, col1, _, col2, _, _ = st.columns(7)

    col1.metric("ROC-AUC", "0.9")
    col2.metric("F1-Score", "0.89")

    st.subheader("MRI Upload")

    col1, col2 = st.columns(2)

    with col1:
        t2_mri = st.file_uploader("Upload T2 MRI", type=["nii", "nii.gz", "nrrd"])

    with col2:
        mask = st.file_uploader("Upload Mask", type=["nii", "nii.gz", "nrrd"])

    st.subheader("Clinical Data Upload")
    col3, col4, col5 = st.columns(3)

    with col3:
        gender = st.radio("Select Gender", ("Male", "Female"))
    with col4:
        manufacturer = st.radio("Select Manufacturer", ("Siemens", "Philips", "GE"))
    with col5:
        age = st.number_input("Enter Age", min_value=18, value=45)
        show_image = st.checkbox("Show MRI with Mask")

    # Button to process the files
    if st.button("Predict ", key="html_button"):
        if t2_mri is not None and mask is not None:
            save_uploaded_file(t2_mri, os.path.join("static", t2_mri.name))
            save_uploaded_file(mask, os.path.join("static", mask.name))
            clinical_data = {
                'age': age,
                'sex': 'F' if gender == 'Female' else 'M',
                'manufacturer': manufacturer
            }
            result = process_files(t2_mri.name, mask.name, clinical_data)

            if result:
                st.write("Malignant lesion (intrahepatic cholangiocarcinoma, hepatocellular carcinoma)")
            else:
                st.write(f"Benign lesion (hepatocellular adenoma, focal nodular hyperplasia)")

            if show_image:
                image_path = "static/show_slice.jpg"  # Ensure this path is correct
                if os.path.exists(image_path):
                    st.image(image_path, caption="MRI with Mask", use_container_width=True)
                    os.remove(image_path)
                else:
                    st.error("Image not found.")
        else:
            st.error("Please upload both files.")


if __name__ == "__main__":
    main()
