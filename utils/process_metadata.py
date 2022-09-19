from curses import meta
import os, glob
from tqdm import tqdm
import pandas as pd
import pydicom as dicom


# From https://www.kaggle.com/code/andradaolteanu/rsna-fracture-detection-dicom-images-explore
def get_observation_data(path):
    '''
    Get information from the .dcm files
    '''

    dataset = dicom.read_file(path)
    
    # Dictionary to store the information from the image
    observation_data = {
        "Rows" : dataset.get("Rows"),
        "Columns" : dataset.get("Columns"),
        "SOPInstanceUID" : dataset.get("SOPInstanceUID"),
        "ContentDate" : dataset.get("ContentDate"),
        "SliceThickness" : dataset.get("SliceThickness"),
        "InstanceNumber" : dataset.get("InstanceNumber"),
        "ImagePositionPatient" : dataset.get("ImagePositionPatient"),
        "ImageOrientationPatient" : dataset.get("ImageOrientationPatient"),
    }
    
    # String columns
    str_columns = ["SOPInstanceUID", "ContentDate", 
                   "SliceThickness", "InstanceNumber"]
    for k in str_columns:
        observation_data[k] = str(dataset.get(k)) if k in dataset else None

    return observation_data


# From https://www.kaggle.com/code/andradaolteanu/rsna-fracture-detection-dicom-images-explore
def get_metadata(train_df, image_dir='train_images'):
    '''
    Retrieves the desired metadata from the .dcm files and saves it into dataframe.
    '''
    
    exceptions = 0
    dicts = []

    for k in tqdm(range(len(train_df))):
        if (k % 100)==0:
            print(f'Iteration: {k}')
            
        dt = train_df.iloc[k, :]

        # Get all .dcm paths for this Instance
        dcm_paths = glob.glob(f"{image_dir}/{dt.StudyInstanceUID}/*")

        for path in dcm_paths:
            try:
                # Get datasets
                dataset = get_observation_data(path)
                dicts.append(dataset)
            except Exception as e:
                exceptions += 1
                continue

    # Convert into df
    columns = [
        "Rows", "Columns", "SOPInstanceUID", "ContentDate", "SliceThickness", 
        "InstanceNumber", "ImagePositionPatient", "ImageOrientationPatient", 
        "SOPInstanceUID", "ContentDate", "SliceThickness", "InstanceNumber"
    ]
    meta_train_data = pd.DataFrame(data=dicts, columns=columns)
    
    # Export information
    # meta_train_data.to_csv("meta_train.csv", index=False)
    
    print(f"Metadata created. Number of total fails: {exceptions}.")
    return meta_train_data


# From https://www.kaggle.com/code/samuelcortinhas/rsna-fracture-detection-in-depth-eda
def clean_metadata(meta_train):
    meta_train_clean = meta_train.drop(['SOPInstanceUID','ImagePositionPatient','ImageOrientationPatient','ImageSize'], axis=1)
    meta_train_clean.rename(columns={"Rows": "ImageHeight", "Columns": "ImageWidth","InstanceNumber": "Slice"}, inplace=True)
    meta_train_clean = meta_train_clean[['StudyInstanceUID','Slice','ImageHeight','ImageWidth','SliceThickness','ImagePositionPatient_x','ImagePositionPatient_y','ImagePositionPatient_z']]
    meta_train_clean.sort_values(by=['StudyInstanceUID','Slice'], inplace=True)
    meta_train_clean.reset_index(drop=True, inplace=True)

    # Export information
    # meta_train_clean.to_csv("meta_train_clean.csv", index=False)
    return meta_train_clean


if __name__ == "__main__":
    train_df = pd.read_csv('train.csv')
    meta_train = get_metadata(train_df, image_dir='train_images')
    meta_train = clean_metadata(meta_train)
    print(meta_train)