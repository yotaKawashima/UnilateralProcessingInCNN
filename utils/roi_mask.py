import os 
import numpy as np 

all_rois = ["all-vertices", 
            "V1v", "V2v", "V3v", "hV4",
            "V1d", "V2d", "V3d", 
            "EBA", "FBA-1", "FBA-2",
            "mTL-bodies", "OFA", "FFA-1", "FFA-2", 
            "mTL-faces", "aTL-faces", "OPA", "PPA", "RSC", 
            "OWFA", "VWFA-1", "VWFA-2", "mfs-words", 
            "mTL-words", 
            "early", "midventral",  "ventral",
            "midlateral" , "lateral",
            "midparietal", "parietal"]

def get_roi_mask(roi, hemisphere, subj_dir):
    """
    Retrieves the Region of Interest (ROI) mask for a given subject, ROI, and hemisphere.

    Args:
        roi (str): The name of the Region of Interest (ROI). Valid options include:
                   "V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", 
                   "mTL-bodies", "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA", 
                   "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words", 
                   "early", "midventral", "midlateral", "midparietal", "ventral", "lateral", 
                   "parietal", "all-vertices".
        hemisphere (str): The hemisphere of the brain ('left' or 'right').
        subj_dir (str): The directory path for the subject's data.

    Returns:
        tuple: A tuple containing:
            - algonauts_roi (numpy.ndarray or None): The ROI mask in the challenge space, 
              or None if the ROI is 'all-vertices'.
            - fsaverage_roi (numpy.ndarray): The ROI mask in the fsaverage space.
    
    Raises:
        ValueError: If an invalid ROI name is provided.
    """
       
    # Define the ROI class based on the selected ROI
    if roi in ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]:
        roi_class = 'prf-visualrois'
    elif roi in ["EBA", "FBA-1", "FBA-2", "mTL-bodies"]:
        roi_class = 'floc-bodies'
    elif roi in ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"]:
        roi_class = 'floc-faces'
    elif roi in ["OPA", "PPA", "RSC"]:
        roi_class = 'floc-places'
    elif roi in ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]:
        roi_class = 'floc-words'
    elif roi in ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]:
        roi_class = 'streams'
    elif roi == 'all-vertices':
        roi_class = 'all-vertices'
    else:
        raise ValueError('Invalid ROI name.')

    # Load the ROI brain surface maps
    fsaverage_roi_class_dir = \
        os.path.join(subj_dir, 'roi_masks',
                     hemisphere[0]+'h.'+roi_class+'_fsaverage_space.npy')

    fsaverage_roi_class = np.load(fsaverage_roi_class_dir)

    if roi != 'all-vertices':
        algonauts_roi_class_dir = \
            os.path.join(subj_dir, 'roi_masks',
                         hemisphere[0]+'h.'+roi_class+'_challenge_space.npy')

        algonauts_roi_class = np.load(algonauts_roi_class_dir)

        roi_map_dir = \
            os.path.join(subj_dir, 'roi_masks',
                         'mapping_'+roi_class+'.npy')
        roi_map = np.load(roi_map_dir, allow_pickle=True).item()

        # Select the vertices corresponding to the ROI of interest
        roi_mapping = list(roi_map.keys())[list(roi_map.values()).index(roi)]
        algonauts_roi = np.asarray(algonauts_roi_class == roi_mapping, dtype=int) # ROI definitions with only the algonauts visual area vertices
        fsaverage_roi = np.asarray(fsaverage_roi_class == roi_mapping, dtype=int) # ROI definitions with all the full brain fsaverage vertices

        return algonauts_roi, fsaverage_roi

    else:
        return None, fsaverage_roi_class


def get_roi_from_vertex_idxs(vertex_idxs, hemisphere, subj_dir):
    """
    Retrieves the Region of Interest (ROI) for given vertex indices for a specific subject and hemisphere.

    Args:
        vertex_idxs (list): List of vertex indices.
        hemisphere (str): The hemisphere of the brain ('left' or 'right').
        subj_dir (str): The directory path for the subject's data.

    Returns:
        dict: A dictionary where the keys are vertex indices and the values are lists of ROIs corresponding to each vertex index.
    """
    # Initialise data storage (key = vertex_id, value = list of rois for the vertex_id)
    rois_given_vertex_idxs = {k:[] for k in vertex_idxs}

    # List of the ROI class
    roi_class_lists = \
        ['prf-visualrois', 'floc-bodies', 'floc-faces', \
         'floc-places', 'floc-words', 'streams']

    for roi_class in roi_class_lists:
        # Load the ROI brain surface maps
        fsaverage_roi_class_dir = \
            os.path.join(subj_dir, 'roi_masks',
                        hemisphere[0]+'h.'+roi_class+'_fsaverage_space.npy')

        fsaverage_roi_class = np.load(fsaverage_roi_class_dir)

        algonauts_roi_class_dir = \
            os.path.join(subj_dir, 'roi_masks',
                            hemisphere[0]+'h.'+roi_class+'_challenge_space.npy')

        algonauts_roi_class = np.load(algonauts_roi_class_dir)

        roi_map_dir = \
            os.path.join(subj_dir, 'roi_masks',
                            'mapping_'+roi_class+'.npy')
        roi_map = np.load(roi_map_dir, allow_pickle=True).item()
        
        # obtain the roi for the vertex_idxs
        code_vertices = algonauts_roi_class[vertex_idxs]
        
        for i, vertex_id in enumerate(vertex_idxs):
            code_this_vertex = code_vertices[i]
            # store roi if code is non-zero  (0 = 'unknown')
            if code_this_vertex != 0: 
                rois_given_vertex_idxs[vertex_id].append(roi_map.get(code_this_vertex))

    return rois_given_vertex_idxs


def get_vertex_idxs_from_rois(rois, subj_dir):
    """
    Retrieves vertex indices and a combined ROI mask for given ROIs for a specific subject.
    Note that vertex indices are the first vertex of the ROI.

    Args:
        rois (list): List of ROIs for which vertex indices are to be retrieved.
        subj_dir (str): The directory path for the subject's data.

    Returns:
        tuple: A tuple containing:
            - vertex_idxs (list): List of vertex indices corresponding to the given ROIs.
            - roi_masks (np.ndarray): A combined ROI mask where each ROI is represented by a unique integer.

    Raises:
        ValueError: If an invalid ROI name is provided.
    """
    
    # initialise list
    vertex_idxs = [] # max(vertex_idxs) = 19003

    for i, roi in enumerate(rois):
        # search verticies for a given roi 
        roi_mask, _ = get_roi_mask(roi, 'left', subj_dir)
        # keep the first vertex from the roi
        vertex_idxs.append(np.where(roi_mask != 0)[0][0])
        
        # create a new mask to visualise the rois in one plot. 
        if i == 0:
            roi_masks = np.zeros(roi_mask.shape)
        roi_masks[np.where(roi_mask != 0)[0]] = i + 1

    # check other roi name for the vertices 
    rois_given_vertex_idxs = \
    get_roi_from_vertex_idxs(vertex_idxs, 'left', subj_dir) 
    print('ROIs for given vertices')
    print(rois_given_vertex_idxs)
    
    return vertex_idxs, roi_masks


def take_mean_within_each_ROI_one_hemi(data, data_description, hemisphere, rois, subj, fmri_dir, model_name, model_layer):
    """
    Calculate the mean value within each Region of Interest (ROI) for one hemisphere.
    Args:
        data (numpy.ndarray): Data array.
        data_description (str): Description of the data (e.g. extraction mode, subtract etc).
        hemisphere (str): The hemisphere to consider ('left' or 'right').
        rois (list): List of ROIs to process.
        subj (int): Subject identifier.
        fmri_dir (str): Directory containing fMRI data.
        model_name (str): Name of the model used.
        model_layer (str): Layer of the model used.
    Returns:
        pandas.DataFrame: DataFrame containing the mean values for each ROI, along with metadata.
    """

    subj_dir = os.path.join(fmri_dir, f'subj{subj:02d}')

    # initialise a list to store the results
    df = []
    for roi in rois:
        # get mask
        if hemisphere == 'left':
            roi_mask, _ = get_roi_mask(roi, 'left', subj_dir)
        else:
            roi_mask, _ = get_roi_mask(roi, 'right', subj_dir)
        
        # store data
        df.append({
            'subject': f'subj-{subj:02d}',
            'mean_corr': np.mean(data[roi_mask!=0]),
            'hemisphere': hemisphere,
            'roi': roi, 
            'model_name' : model_name, 
            'model_layer': model_layer, 
            'extraction_mode': data_description})
    return pd.DataFrame(df)