import os
import streamlit as st

from feature_extracting import FeatureExtractor
from preprocessing import Preprocessor
from predicting import Predictor



# Function to process the uploaded files
def process_files(image_name: str, mask_name: str, clinical_data: dict):
    model = 'liver_t2w_xgboost'
    preprocessor = Preprocessor(mri_modality='T2')
    feature_extractor = FeatureExtractor(model, desired_order_bool=False)
    predictor = Predictor(model)

    image_path = os.path.join("uploads", image_name)
    mask_path = os.path.join("uploads", mask_name)

    image, mask = preprocessor.preprocessing_step(image_path, mask_path, normalize=True, resample=False)

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

    _, _,  col1, _, col2, _, _ = st.columns(7)

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

    # Button to process the files
    if st.button("Process Files"):
        if t2_mri is not None and mask is not None:
            save_uploaded_file(t2_mri, os.path.join("uploads", t2_mri.name))
            save_uploaded_file(mask, os.path.join("uploads", mask.name))
            clinical_data = {
                'age': age,
                'sex': 'F' if gender == 'Female' else 'M',
                'manufacturer': manufacturer
            }
            result = process_files(t2_mri.name, mask.name, clinical_data)
            st.write(f"The output is: {result}")
        else:
            st.error("Please upload both files.")


if __name__ == "__main__":
    main()
