import os 
import numpy as np
from nilearn import datasets
from nilearn import plotting
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.roi_mask import get_roi_mask, all_rois 

def plot_brainmap(data_to_plot, roi, map_type, hemisphere, subj_dir, title=None):
	"""
    Plots an interactive brain surface map for a given subject, ROI, and hemisphere.

    Args:
        data_to_plot (array-like or None): 
			Data to be plotted (e.g. fMRI data). If None, the ROI mask will be plotted instead.
        roi (str): The name of the Region of Interest (ROI). Valid options include:
                   "V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", 
                   "mTL-bodies", "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA", 
                   "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words", 
                   "early", "midventral", "midlateral", "midparietal", "ventral", "lateral", 
                   "parietal", "all-vertices".
        map_type (str): The type of brain map to plot ('infl', 'pial', 'sphere', 'white', 'flat']).
        hemisphere (str): The hemisphere of the brain ('left' or 'right').
        subj_dir (str): The directory path for the subject's data.
        title (str, optional): The title of the plot. If None, a default title will be generated.

    Returns:
        view: An interactive brain surface map view object.
    """
	algonauts_roi, fsaverage_roi = get_roi_mask(roi, hemisphere, subj_dir)

	if data_to_plot is None:
		fsaverage_response = fsaverage_roi
		cmap = 'cool'
		colorbar = False
	else:
		fsaverage_response = np.zeros(len(fsaverage_roi))
		if roi != 'all-vertices':
			# We need to find which fsaverage vertices correspond to the algonauts vertices, and fill in data there.
			fsaverage_response[np.where(fsaverage_roi)[0][:len(np.where(algonauts_roi)[0])]] = \
				data_to_plot[np.where(algonauts_roi)[0]]
		else:
			fsaverage_response[np.where(fsaverage_roi)[0]] = data_to_plot
		cmap = 'cold_hot'
		colorbar = True

	if title is None:
		title = roi+', '+hemisphere+' hemisphere'

	# Create the interactive brain surface map
	fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
	view = plotting.view_surf(
		surf_mesh=fsaverage[map_type+'_'+hemisphere],
		surf_map=fsaverage_response,
		bg_map=fsaverage['sulc_'+hemisphere],
		threshold=1e-14,
		cmap=cmap,
		colorbar=True,
		title=roi+', '+hemisphere+' hemisphere'
		)
	return view


def make_rois_encoding_plot(prediction_df: pd.DataFrame, hue, title=None, plot_range=None):
	"""
    Display a bar plot showing the mean correlation for different Regions of Interest (ROIs) 
    across hemispheres.

    Args:
        prediction_df (pd.DataFrame): A DataFrame containing the prediction results. 
                                      It should have the following columns:
                                      - 'roi': The Region of Interest.
                                      - 'mean_corr': The mean correlation value.
		hue (str): The column name in the DataFrame to use for coloring the bars.
		title (str, optional): The title of the plot. If None, a default title will be generated.
		plot_range (list of float, optional): The minimum and maximum value for the barplot. 
						If None, pyplot automatically determines the minimum value.
		
    Returns:
        None: This function does not return any value. It displays a bar plot.
    """
	fig_barplot = sns.barplot(x='roi', y='mean_corr', hue=hue,
						      errorbar='se', data=prediction_df)
	plt.xlabel('ROI')
	plt.ylabel('Mean correlation within each ROI')
	if hue != None:
		plt.legend(title=hue) 
	plt.xticks(rotation=90)
	num_participants = len(prediction_df['subject'].unique())
	if plot_range != None:
		plt.ylim(plot_range[0], plot_range[1])
	if title == None:
		plt.title(f'Mean correlation from {num_participants} participants \n (bar = mean, error bar = std across participants)')
	else: 
		plt.title(title)
	plt.show()
	return


def make_scatter_recorded_predicted_fMRI(prediction_single_participant_df: pd.DataFrame, model_name, model_layer):
	"""
    Creates scatter plots comparing recorded and predicted left hemisphere fMRI data for a single participant.

    Args:
        prediction_single_participant_df (pd.DataFrame): DataFrame containing prediction results for a single participant.
			It should have the following columns:
			- 'extraction_mode': The mode of feature extraction.
			- 'model_name': The name of the model.
			- 'model_layer': The layer of the model.
			- 'vertex_idxs': The indices of the vertices.
			- 'lh_fmri_test': The recorded fMRI test data for the left hemisphere. (only specified vertices)
			- 'lh_fmri_test_pred': The predicted fMRI test data for the left hemisphere. (only specified verticies)
			- 'lh_corrs': The correlation values for the left hemisphere. (all verticies)
        model_name (str): The name of the model to be used for filtering the DataFrame.
        model_layer (str): The layer of the model to be used for filtering the DataFrame.

    Returns:
        None: This function does not return any value. It displays scatter plots comparing real and predicted fMRI data.
    """
	extraction_mode_list = \
	prediction_single_participant_df['extraction_mode'].unique()
	
	vertex_idxs = prediction_single_participant_df['vertex_idxs'][0]

	num_vertecices = len(vertex_idxs)
	fig, axes = plt.subplots(1, num_vertecices, figsize=(20, 6))

	for i_ax, i_vertex in enumerate(vertex_idxs):
		# loop through all modes
		for i_extraction_mode in range(len(extraction_mode_list)):
			extraction_mode = extraction_mode_list[i_extraction_mode]
			data_this_cond = \
			prediction_single_participant_df[(prediction_single_participant_df.extraction_mode == extraction_mode) & 
												(prediction_single_participant_df.model_name == model_name) & 
												(prediction_single_participant_df.model_layer == model_layer)]

			# scatter plot 
			# note that lh_corrs contains data for all verticies, 
			# but fmri_test and fmri_test_pred contain data for only the specified verticies (vertex_idxs).
			sns.regplot(x=data_this_cond['lh_fmri_test'].item()[:, i_ax], 
			            y= data_this_cond['lh_fmri_test_pred'].item()[:, i_ax], 
			            ci=None,
			            label=f"{extraction_mode} (Pearson={data_this_cond['lh_corrs'].item()[i_vertex]:.2f})",
               		  ax=axes[i_ax])

			# axes[i_ax].scatter(data_this_cond['lh_fmri_test'].item()[:, i_ax], 
			# 				data_this_cond['lh_fmri_test_pred'].item()[:, i_ax], 
			# 				label=f"{extraction_mode} (Pearson={data_this_cond['lh_corrs'].item()[i_vertex]:.2f})")

		# figure settings 
		#x_min, x_max = min(data_this_cond['lh_fmri_test'].item()[:, i_ax]), max(data_this_cond['lh_fmri_test'].item()[:, i_ax])
		#y_min, y_max = min(data_this_cond['lh_fmri_test_pred'].item()[:, i_ax]), max(data_this_cond['lh_fmri_test_pred'].item()[:, i_ax])
		#min_val = min(x_min, y_min)
		#max_val = max(x_max, y_max)
		#axes[i_ax].set_xlim(min_val, max_val)
		#axes[i_ax].set_ylim(min_val, max_val)
		axes[i_ax].set_box_aspect(1)
		axes[i_ax].set_xlabel('Recorded fMRI data')
		axes[i_ax].set_title(f'vertex: {i_vertex}')
		axes[i_ax].legend(loc='upper left', bbox_to_anchor=(1, 0.5)) 
		axes[i_ax].set_ylabel('Predicted fMRI data')

	# Add title
	fig.tight_layout()
	fig.suptitle(f'Predicting the left hemisphere fMRI data by {model_layer} layer in {model_name}')
	fig.show()
	
	return 
