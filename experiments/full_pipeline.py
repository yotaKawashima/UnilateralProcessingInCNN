import os
import pandas as pd
import models
from experiments import encoding_model, dataset_preparation
import models.vision_dnn

def single_subj_all_cond(fmri_dir, imgs_paths, device, batch_size, 
                         subj, model_layer_info, vertex_idxs, 
                         extraction_mode_list=['full', 'left', 'right'],
                         seed=0, train_perc=90) -> pd.DataFrame:    
    """ Analyse data for a single subject using all conditions.
    1. Model set-up
    2. Feature extraction 
    3. Encoding model fitting and prediction.
    Note that we store real and predicted fMRI data for the specified vertcies in the only left hemisphere. 

    Args:
        fmri_dir (str): Directory containing fMRI data for all subjects.
        imgs_paths (list): List of image paths.
        device(torch.device): Device to be used for training.
        batch_size (int): Batch size for the dataloader.
        subj (str): Subject number.
        model_layer_info (list of dict): list of neural network models and layer name to be used.
        vertex_idxs (list): list of vertex indices to be stored.
        extraction_mode_list (list, optional): list of extraction modes to be used. Default is ['full', 'left', 'right'].
        seed (int, optional): Random seed for reproducibility. Default is 0.
        train_perc (int, optional): Percentage of data to be used for training. Default is 90.

    Returns:
        pd.DataFrame: A DataFrame containing the results.

    Raises:
        ValueError: If an invalid model name is provided.
    """

    # convert model_layer_info to a dataframe
    model_layer_info = pd.DataFrame(model_layer_info)

    # initialise dataframe to store data.
    prediction_single_participant_df = []

    # loop through two models.
    for model_name in model_layer_info.model_name.unique():
        # get pre-trained model
        if model_name == 'effNet':
            model, transforms = models.vision_dnn.get_effnet_b0(device)
        elif model_name == 'effNetWithoutPretraining':
            model, transforms = models.vision_dnn.get_effnet_b0(device, weights=None)
        elif model_name == 'effNetTrained':
            model, transforms = models.vision_dnn.get_effnet_b0(device)
        elif model_name == 'resNet18':
            model, transforms = models.vision_dnn.get_resnet18(device)
        elif model_name == 'resNet18WithoutPretraining':
            model, transforms = models.vision_dnn.get_resnet18(device, weights=None)
        else: 
            raise ValueError(f"Invalid model name: {model_name}")

        # get dataloaders for the model
        dataloader_train, dataloader_test, idxs_train, idxs_test = \
            dataset_preparation.get_dataloaders(imgs_paths, transforms,
                                                device, batch_size,
                                                train_perc=train_perc,
                                                seed=seed)

        # set up feature extractor
        feature_extractor = \
            models.feature_extraction.FeatureExtractor(model,
                                                       dataloader_train,
                                                       dataloader_test,
                                                       reduction=True,
                                                       n_components=batch_size)
        # loop through all extraction mode
        for extraction_mode in extraction_mode_list:
            # loop through all model layers
            for model_layer in model_layer_info[model_layer_info.model_name == model_name].model_layer:
                # specify model layer and extraction_mode (full, right or left)
                feature_extractor.set_feature_extractor(model_layer, extraction_mode=extraction_mode)
                # extract features for each dataset
                print("training dataset")
                features_train = feature_extractor.extract_features('train')
                print("testing dataset")
                features_test = feature_extractor.extract_features('test')

                # try encoding model for the given participant
                subj_dir = os.path.join(fmri_dir, f'subj{subj:02d}')

                lh_corrs, rh_corrs, _, _, lh_fmri_test, _, lh_fmri_test_pred, _ = \
                    encoding_model.fit_and_predict(fmri_dir, subj,
                                                   features_train, features_test,
                                                   idxs_train, idxs_test)
                
                # store data. (Note that we store lh_fmri_test and lh_fmri_test_pred
                # from the specified vertex.)
                prediction_single_participant_df.append({
                            'subject': f'subj-{subj:02d}',
                            'model_name' : model_name, 
                            'model_layer': model_layer, 
                            'extraction_mode': extraction_mode,
                            'lh_corrs': lh_corrs,
                            'rh_corrs': rh_corrs, 
                            'vertex_idxs': vertex_idxs,
                            'lh_fmri_test': lh_fmri_test[:, vertex_idxs], 
                            'lh_fmri_test_pred': lh_fmri_test_pred[:, vertex_idxs]
                        })
                # detele features and prediction from the previous layer
                del features_train, features_test, \
                    lh_corrs, rh_corrs, lh_fmri_test, lh_fmri_test_pred

        # once you finish all analysis for this NN model, you delete models,
        # feature_extractor, and dataloaders.
        del model, transforms, feature_extractor, \
            dataloader_train, dataloader_test, idxs_train, idxs_test

    return pd.DataFrame(prediction_single_participant_df)


def all_subj_all_cond(fmri_dir, imgs_paths, device, batch_size,
                      model_layer_info, rois,
                      extraction_mode_list=['full', 'left', 'right'],
                      seed=0, train_perc=90) -> pd.DataFrame:  
    """ Analyse data for all subjects using all conditions.
    1. Model set-up
    2. Feature extraction 
    3. Encoding model fitting and prediction.

    Args:
        fmri_dir (str): Directory containing fMRI data for all subjects.
        imgs_paths (list): List of image paths.
        device(torch.device): Device to be used for training.
        batch_size (int): Batch size for the dataloader.
        rois (list): List of regions of interest (ROIs) to be analyzed.
        model_layer_info (list of dict): list of neural network models and layer name to be used.
        vertex_idxs (list): list of vertex indices to be stored.
        extraction_mode_list (list, optional): list of extraction modes to be used. Default is ['full', 'left', 'right'].
        seed (int, optional): Random seed for reproducibility. Default is 0.
        train_perc (int, optional): Percentage of data to be used for training. Default is 90.

    Returns:
        pd.DataFrame: A DataFrame containing the results.
    
    Raises: 
        ValueError: If an invalid model name is provided.
    """   
    # convert model_layer_info to a dataframe
    model_layer_info = pd.DataFrame(model_layer_info)

    # initialise dataframe to store data.
    prediction_df = None

    # loop through two models.
    for model_name in model_layer_info.model_name.unique():
        # get pre-trained model
        if model_name == 'effNet':
            model, transforms = models.vision_dnn.get_effnet_b0(device)
        elif model_name == 'effNetWithoutPretraining':
            model, transforms = models.vision_dnn.get_effnet_b0(device, weights=None)
        elif model_name == 'effNetTrained':
            model, transforms = models.vision_dnn.get_effnet_b0(device)
        elif model_name == 'resNet18':
            model, transforms = models.vision_dnn.get_resnet18(device)
        elif model_name == 'resNet18WithoutPretraining':
            model, transforms = models.vision_dnn.get_resnet18(device, weights=None)
        else: 
            raise ValueError(f"Invalid model name: {model_name}")
        
        # get dataloaders for the model
        dataloader_train, dataloader_test, idxs_train, idxs_test = \
            dataset_preparation.get_dataloaders(imgs_paths, transforms,
                                                device, batch_size,
                                                train_perc=train_perc,
                                                seed=seed)

        # set up feature extractor
        feature_extractor = \
            models.feature_extraction.FeatureExtractor(model,
                                                       dataloader_train,
                                                       dataloader_test,
                                                       reduction=True,
                                                       n_components=batch_size)
        # loop through all extraction mode
        for extraction_mode in extraction_mode_list:
            # loop through all model layers
            for model_layer in model_layer_info[model_layer_info.model_name == model_name].model_layer:
                # specify model layer and extraction_mode (full, right or left)
                feature_extractor.set_feature_extractor(model_layer, extraction_mode=extraction_mode)
                # extract features for each dataset
                print("training dataset")
                features_train = feature_extractor.extract_features('train')
                print("testing dataset")
                features_test = feature_extractor.extract_features('test')

                # try encoding model for each participant
                prediction_df_this_condition = \
                    encoding_model.fit_and_predict_all_subj(fmri_dir,
                                                            features_train, features_test,
                                                            idxs_train, idxs_test,
                                                            rois,
                                                            model_name,
                                                            model_layer, extraction_mode)

                # concatenate dataframe
                if prediction_df is None:
                    prediction_df = prediction_df_this_condition
                else:
                    prediction_df = pd.concat([prediction_df, prediction_df_this_condition])

                # detele features and prediction from the previous layer
                del features_train, features_test, prediction_df_this_condition

        # once you finish all analysis for this NN model, you delete models,
        # feature_extractor, and dataloaders.
        del model, transforms, feature_extractor, \
            dataloader_train, dataloader_test, idxs_train, idxs_test
        
    return pd.DataFrame(prediction_df)