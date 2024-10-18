import os 
import numpy as np
import pandas as pd
from tqdm import tqdm 
from sklearn.linear_model import LinearRegression
from sklearn.base import clone
from scipy.stats import pearsonr
from utils.roi_mask import get_roi_mask 

def fit_and_predict(fmri_dir, subj,
                    features_train, features_test,
                    idxs_train, idxs_test,
                    regression=LinearRegression()):
    """
    Fits a regression model to fMRI data and predicts the test data for a given subject.

    Args:
        fmri_dir (str): Directory containing fMRI data for all subjects.
        subj (int): Subject number.
        features_train (array-like): Training features for the model.
        features_test (array-like): Testing features for the model.
        idxs_train (array-like): Indices for training data.
        idxs_test (array-like): Indices for testing data.
        regression (object, optional): A regression model instance from scikit-learn. Default is LinearRegression().

    Returns:
        tuple: A tuple containing:
            - lh_corrs (np.ndarray): Correlation coefficients for the left hemisphere.
            - rh_corrs (np.ndarray): Correlation coefficients for the right hemisphere.
            - reg_lh (object): Fitted regression model for the left hemisphere.
            - reg_rh (object): Fitted regression model for the right hemisphere.
            - lh_fmri_test (np.ndarray): Ground-truth fMRI data for the left hemisphere.
            - rh_fmri_test (np.ndarray): Ground-truth fMRI data for the right hemisphere.
            - lh_fmri_test_pred (np.ndarray): Predicted fMRI data for the left hemisphere.
            - rh_fmri_test_pred (np.ndarray): Predicted fMRI data for the right hemisphere.
    """

    subj_dir = os.path.join(fmri_dir, f'subj{subj:02d}')

    # Load fmri for both hemispheres of the subject
    lh_fmri = np.load(os.path.join(subj_dir, 'lh_training_fmri.npy'))
    rh_fmri = np.load(os.path.join(subj_dir, 'rh_training_fmri.npy'))

    # Make train/test splits for the fMRI data for the two hemispheres
    lh_fmri_train = lh_fmri[idxs_train]
    lh_fmri_test = lh_fmri[idxs_test]
    rh_fmri_train = rh_fmri[idxs_train]
    rh_fmri_test = rh_fmri[idxs_test]
    del lh_fmri, rh_fmri # Delete unused fMRI data to save memory

    # Make new instances of the linear reg class passed as input
    reg_lh = clone(regression) # This makes a new instance of the regression
    reg_rh = clone(regression) 

    # Fit linear regressions on the training data
    reg_lh = reg_lh.fit(features_train, lh_fmri_train)
    reg_rh = reg_rh.fit(features_train, rh_fmri_train) 

    # Use fitted linear regressions to predict the validation fMRI data
    lh_fmri_test_pred = reg_lh.predict(features_test)
    rh_fmri_test_pred = reg_rh.predict(features_test)
    
    # Correlate predicted and ground-truth values
    # initialise test_corrs to store correlations for each fMRI vertex.
    lh_corrs = np.zeros(lh_fmri_test.shape[1])
    rh_corrs = np.zeros(rh_fmri_test.shape[1])

    # Correlate each predicted fMRI vertex with the corresponding ground truth vertex
    max_num_vertices = max(lh_fmri_test.shape[1], rh_fmri_test.shape[1])
    for v in range(max_num_vertices):
        if lh_fmri_test.shape[1] > rh_fmri_test.shape[1]:
            try:
                lh_corrs[v] = pearsonr(lh_fmri_test_pred[:,v], lh_fmri_test[:,v])[0]
                rh_corrs[v] = pearsonr(rh_fmri_test_pred[:,v], rh_fmri_test[:,v])[0]
            except IndexError:
                continue
        else:
            try:
                rh_corrs[v] = pearsonr(rh_fmri_test_pred[:,v], rh_fmri_test[:,v])[0]
                lh_corrs[v] = pearsonr(lh_fmri_test_pred[:,v], lh_fmri_test[:,v])[0]
            except IndexError:
                continue

    return lh_corrs, rh_corrs, reg_lh, reg_rh, lh_fmri_test, rh_fmri_test, lh_fmri_test_pred, rh_fmri_test_pred


def fit_and_predict_all_subj(fmri_dir,
                             features_train, features_test,
                             idxs_train, idxs_test,
                             rois, model_name, model_layer, extraction_mode) -> pd.DataFrame:    
    """ Fits a linear regression model and predicts fMRI data for all subjects.

    Args:
        fmri_dir (str): Directory containing fMRI data for all subjects.
        features_train (array-like): Training features for the model.
        features_test (array-like): Testing features for the model.
        idxs_train (array-like): Indices for training data.
        idxs_test (array-like): Indices for testing data.
        rois (list): List of regions of interest (ROIs) to be analyzed.
        model_name (str): the name of the neural network model.  
        model_layer (str): The layer of the model to extract features from.
        extraction_mode (str): The mode of feature extraction ('full', 'left' or 'right').

    Returns:
        pd.DataFrame: A DataFrame containing the mean correlation results 
            for each subject, hemisphere, and ROI. Columns include 
            'subject', 'mean_corr', 'hemisphere', and 'roi'.
    """
    # initialise a list to store the results
    prediction_df = []

    # loop over all subjects
    num_subj = len(os.listdir(fmri_dir))
    print('Fitting and predicting for each subject: Start')
    for subj in tqdm(range(1, 1+num_subj)):

        subj_dir = os.path.join(fmri_dir, f'subj{subj:02d}')
        
        # fit and predict for the current subject
        lh_corrs, rh_corrs, _, _, _, _, _, _ = fit_and_predict(fmri_dir, subj,
                                                   features_train, features_test,
                                                   idxs_train, idxs_test)
        
        # summary for each ROI 
        for roi in rois:
            roi_lh_mask, _ = get_roi_mask(roi, 'left', subj_dir)
            roi_rh_mask, _ = get_roi_mask(roi, 'right', subj_dir)

            prediction_df.append({
                'subject': f'subj-{subj:02d}',
                'mean_corr': np.mean(lh_corrs[roi_lh_mask!=0]),
                'hemisphere': 'left',
                'roi': roi, 
                'model_name' : model_name, 
                'model_layer': model_layer, 
                'extraction_mode': extraction_mode

            })
            prediction_df.append({
                'subject': f'subj-{subj:02d}',
                'mean_corr': np.mean(rh_corrs[roi_rh_mask!=0]),
                'hemisphere': 'right',
                'roi': roi, 
                'model_name' : model_name, 
                'model_layer': model_layer, 
                'extraction_mode': extraction_mode

            })
    
    print('Fitting and predicting for each subject: End')    
    # convert to pandd dataframe
    prediction_df = pd.DataFrame(prediction_df)

    return prediction_df