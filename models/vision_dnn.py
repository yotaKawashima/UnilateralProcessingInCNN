import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights, efficientnet_b0, EfficientNet_B0_Weights
from torchinfo import summary

# from tqdm import tqdm
# from torch.utils.data import DataLoader, Dataset
# from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
# from sklearn.decomposition import IncrementalPCA, PCA
# from sklearn.linear_model import LinearRegression
# from sklearn.base import clone


def get_effnet_b0(device, weights=EfficientNet_B0_Weights.DEFAULT):
    """ Get the EfficientNet-B0 model and its preprocessing transforms.
    Args:
        device (str): device where you put model on 
        weights (EfficientNetB0_Weights, optional): Pre-trained weights for the EfficientNet-B0 model. 
            Defaults to EfficientNetB0_Weights.DEFAULT.
    Returns:
        tuple: A tuple containing the EfficientNet-B0 model and its preprocessing transforms.
            - model (torch.nn.Module): The EfficientNet-B0 model.
            - transforms (callable): The preprocessing transforms associated with the model's weights.
    """
    # instantiate model
    model = efficientnet_b0(weights=weights)

    # the weights contain the preprocessing transforms! Very handy.
    if weights == None:
        # use the same transforms as the default weights for not-trained model
        transforms = EfficientNet_B0_Weights.DEFAULT.transforms()
    else: 
        transforms = weights.transforms()

    model.to(device)
    
    # use model in evaluation mode 
    model.eval()
    
    return model, transforms


def get_resnet18(device, weights=ResNet18_Weights.DEFAULT):
    """ Get the ResNet-18 model and its preprocessing transforms.
    Args:
        device (str): device where you put model on 
        weights (ResNet18_Weights, optional): Pre-trained weights for the ResNet-18 model. 
            Defaults to ResNet18_Weights.DEFAULT.
    Returns:
        tuple: A tuple containing the ResNet-18 model and its preprocessing transforms.
            - model (torch.nn.Module): The ResNet-18 model.
            - transforms (callable): The preprocessing transforms associated with the model's weights.
    """
    # instantiate model
    model = resnet18(weights=weights)

    # the weights contain the preprocessing transforms! Very handy.
    if weights == None:
        # use the same transforms as the default weights for not-trained model
        transforms = ResNet18_Weights.DEFAULT.transforms()
    else: 
        transforms = weights.transforms()


    # put model on GPU if possible
    model.to(device)

    # use model in evaluation mode 
    model.eval()
    
    return model, transforms


def show_model_summary(model, input_size=(32, 3, 224, 224)):
    """ Get a summary of the model architecture.
    Args:
        model (torch.nn.Module): The model to summarize.
        input_size (tuple, optional): 
            The input size for the model. Defaults to (32, 3, 224, 224).
    Returns:
        torchinfo.summary.Summary: A summary of the model architecture.
    """
    # get a summary of the model architecture
    summary_model = summary(model=model,
                            input_size=input_size,
                            col_names=["input_size", "output_size", "num_params", "trainable"],
                            col_width=20,
                            row_settings=["var_names"]
    )
    
    return print(summary_model)