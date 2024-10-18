import torch
import numpy as np
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm
from torchvision.models.feature_extraction import create_feature_extractor

# helper function to calculate the ceiling division
def ceildiv(a, b):
    return -(a // -b)

class FeatureExtractor:
    def __init__(self, model,
                 dataloader_train, dataloader_test, 
                 reduction=True, n_components=50):
        """
        Args: 
            model (torch.nn.Module): The pre-trained model.
            dataloader_train (torch.utils.data.DataLoader): 
                A PyTorch DataLoader that provides batches of training data.
            dataloader_test (torch.utils.data.DataLoader):
                A PyTorch DataLoader that provides batches of testing data.
            reduction (bool, optional): 
                Whether to apply PCA for dimensionality reduction. Default is True.
            n_components (int, optional): 
                The number of principal components to compute. Default is 50.
        """
        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.set_reduction(reduction, n_components) 


    def set_reduction(self, reduction, n_components=50):
        self.reduction = reduction
        if reduction == True: 
            self.n_components = n_components
        else: 
            self.n_components = None
        return

    def set_feature_extractor(self, model_layer, extraction_mode):
        """
        Sets the feature extractor to a new model layer.

        Args:
            model_layer (str): Layer name to extract features from.
            extraction_mode (str): 
                The mode of feature extraction. 
                Valid options are 'full', 'left', or 'right'.
        """
        self.model_layer = model_layer
        self.extraction_mode = extraction_mode
        self.feature_extractor = \
                create_feature_extractor(self.model, return_nodes=[model_layer])
        
        # every time you set a new feature extractor, set pca = None.
        # This is to avoid using pca trained for another layer. 
        self.pca = None
        print(f'layer: {self.model_layer}, extraction_mode: {self.extraction_mode}. PCA initialized!')
        print("Make sure that the layer's output is in a form of (batch, input) or (batch, channel, height, width)!")
        return 
        
    def fit_pca(self):
        """
        Fits a PCA model for dimension reduction on features.
        Returns:
            IncrementalPCA: A fitted IncrementalPCA model.
        """
        batch_size = next(iter(self.dataloader_train)).shape[0]

        # Define PCA parameters
        pca = IncrementalPCA(self.n_components, batch_size=batch_size)

        # Fit PCA to batch
        print('Fitting PCA: Start')
        for _, d in tqdm(enumerate(self.dataloader_train), total=len(self.dataloader_train)):
            if d.shape[0] < self.n_components:
                # the last batch can be smaller. If it is smaller than n_components,
                # PCA cannot run so we skip
                continue
            # Extract features
            ft = self._extract_feature_each_input(d)
            # Fit PCA to batch
            pca.partial_fit(ft)
            
        # set self.pca = pca 
        print('Fitting PCA: End')
        self.pca = pca
        return 

    def extract_features(self, dataset_type):
        """
        Extracts neural network activation patterns (features) 
        and optionally applies PCA for dimensionality reduction.

        Args:           
            dataset_type (str): 
            The type of dataset to extract features from. 
            Valid options are 'train' or 'test'.

        Returns:
            np.array: A numpy array containing the extracted features.

        Raises:
            ValueError: If the dataset_type is invalid.
        """
        # set dataloader and fit pca depending on dataset_type 
        if dataset_type=='train':
            dataloader = self.dataloader_train
            if self.reduction==True:
                self.fit_pca()
        elif dataset_type=='test':
            dataloader = self.dataloader_test
        else:
            raise ValueError('Invalid datasets_type. Please select either "train" or "test".')
        
        # initialise list to store features
        features = []
        print('Extracting features: Start')
        for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Extract features
            ft = self._extract_feature_each_input(d)
            
            # Apply PCA transform
            if self.reduction==True:
                ft = self.pca.transform(ft)
            features.append(ft)
        
        print('Extracting features: End')
        return np.vstack(features)


    def _extract_feature_each_input(self, input_data):
        """
        Extracts and processes features for a given input data based on the specified extraction_mode.

        Args:
            input_data (torch.Tensor): The input data from which features are to be extracted.

        Returns:
            np.array: The processed feature array .

        Raises:
            ValueError: If the specified layer has width=1 and cannot be split along the width.
            ValueError: If the specified layer does not have a valid dimension.
            ValueError: If an invalid extraction_mode is specified.
        """        
        # Extract features
        ft = self.feature_extractor(input_data)

        # Process the extracted features based on the extraction_mode
        if self.extraction_mode =='full':
            # Flatten the features
            ft = torch.flatten(ft[self.model_layer], start_dim=1)
        elif self.extraction_mode =='left':                
            # for the left hemisphere, we will take the left half of the tensor
            ft = ft[self.model_layer]
            if ft.dim() == 4: 
                # When tensor is in a form of (batch, channel, height, width).
                # e.g. conv layers
                if ft.shape[3] == 1:
                    # if the tensor is 1D, we cannot split it along the width
                    raise ValueError('The specificed layer has width=1. Cannot split along the width.')
                else:                        
                    ft = torch.flatten(ft[:,:,:,:ft.shape[3]//2], start_dim=1)
            elif ft.dim() == 2: 
                # When tensor is in a form of (batch, input).
                # e.g. linear layers
                ft = ft[:,:ft.shape[1]//2] 
            else:
                raise ValueError('The specified layer does not have a valid dimension.')

        elif self.extraction_mode == 'right':
            # for the right hemisphere, we will take the right half of the tensor
            ft = ft[self.model_layer]
            if ft.dim() == 4: 
                # When tensor is in a form of (batch, channel, height, width).
                # e.g. conv layers
                if ft.shape[3] == 1:
                    # if the tensor is 1D, we cannot split it along the width
                    raise ValueError('The specificed layer has width=1. Cannot split along the width.')
                else:                        
                    ft = torch.flatten(ft[:,:,:, ceildiv(ft.shape[3], 2):], start_dim=1)
            elif ft.dim() == 2: 
                # When tensor is in a form of (batch, input).
                # e.g. linear layers
                ft = ft[:, ceildiv(ft.shape[1], 2):] 
            else:
                raise ValueError('The specified layer does not have a valid dimension.')
        else: 
            raise ValueError('Invalid mode. Please select either "full", "left", or "right".')

        return ft.cpu().detach().numpy()
