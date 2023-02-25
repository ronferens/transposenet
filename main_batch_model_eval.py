"""
Entry point training and testing TransPoseNet
"""
import argparse
import pandas as pd
import torch
import numpy as np
import json
import logging
from util import utils, rot_utils
import time
from datasets.CameraPoseDataset import CameraPoseDataset
from models.pose_regressors import get_model
from os import listdir
from os.path import join, splitext
import plotly
import plotly.graph_objects as go
import hydra
from omegaconf import OmegaConf


def sort_models_name(models_list):
    models_list = sorted(models_list, key=lambda x: (len(x), x))
    if '_final' in models_list[0] and len(models_list) > 1:
        models_list = models_list[1:] + models_list[:1]
    return models_list


@hydra.main(version_base=None, config_path="config", config_name="test")
def main(cfg) -> None:
    assert cfg.inputs.models_path is not None, 'You must specify the models_path'
    utils.init_logger(outpath=cfg.inputs.models_path, suffix='_batch_eval')

    # Record execution details
    logging.info("Start {} with {}".format(cfg.inputs.model_name, cfg.inputs.mode))
    logging.info("Using dataset: {}".format(cfg.inputs.dataset_path))
    logging.info("Using labels file: {}".format(cfg.inputs.labels_file))

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(numpy_seed)

    # Extract all checkpoints to evaluate
    all_model_files = [join(cfg.inputs.models_path, f)
                       for f in listdir(cfg.inputs.models_path) if splitext(f)[-1] == '.pth']

    # Filter out the models
    if cfg.inputs.models_to_evaluate is not None:
        models_to_evaluate = [l for l in cfg.inputs.models_to_evaluate.split(',')]
        models_to_eval = []
        for m in all_model_files:
            experiment_suffix = m.split('_')[-1]
            for e in models_to_evaluate:
                if e in experiment_suffix:
                    models_to_eval.append(m)
    else:
        models_to_eval = all_model_files

    # Sorting the listed models
    models_to_eval = sort_models_name(models_to_eval)

    batch_eval_results = []

    # Go over all models
    for checkpoint_path in models_to_eval:
        # Create the model
        model_config = OmegaConf.to_container(cfg[cfg.inputs.model_name])
        model = get_model(cfg.inputs.model_name, cfg.inputs.backbone_path, model_config)
        model = torch.nn.DataParallel(model, device_ids=cfg.general.devices_ids)
        model.cuda()

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info("Initializing from checkpoint: {}".format(checkpoint_path))

        # Set to eval mode
        model.eval()

        # Set the dataset and data loader
        transform = utils.test_transforms.get('baseline')
        dataset = CameraPoseDataset(cfg.inputs.dataset_path, cfg.inputs.labels_file, transform)
        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': cfg.general.n_workers}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        stats = np.zeros((len(dataloader.dataset), 3))

        with torch.no_grad():
            for i, minibatch in enumerate(dataloader, 0):
                for k, v in minibatch.items():
                    minibatch[k] = v.cuda()

                gt_pose = minibatch.get('pose').to(dtype=torch.float32)

                # Forward pass to predict the pose
                tic = time.time()
                est_pose = model(minibatch).get('pose')
                toc = time.time()

                if cfg[cfg.inputs.model_name]['rot_mode'] == "4D_norm":
                    est_rot = rot_utils.compute_rotation_matrix_from_quaternion(est_pose[:, 3:])  # b*3*3
                elif cfg[cfg.inputs.model_name]['rot_mode'] == "6D_GM":
                    est_rot = rot_utils.compute_rotation_matrix_from_ortho6d(est_pose[:, 3:])  # b*3*3
                elif cfg[cfg.inputs.model_name]['rot_mode'] == "9D_SVD":
                    est_rot = rot_utils.symmetric_orthogonalization(est_pose[:, 3:])  # b*3*3
                elif cfg[cfg.inputs.model_name]['rot_mode'] == "10D":
                    est_rot = rot_utils.compute_rotation_matrix_from_10d(est_pose[:, 3:])  # b*3*3
                elif cfg[cfg.inputs.model_name]['rot_mode'] == "3D_Euler":
                    est_rot = rot_utils.compute_rotation_matrix_from_euler(est_pose[:, 3:])  # b*3*3
                elif cfg[cfg.inputs.model_name]['rot_mode'] == "4D_Axis":
                    est_rot = rot_utils.compute_rotation_matrix_from_axisAngle(est_pose[:, 3:])  # b*3*3
                else:
                    raise NotImplementedError

                # Evaluate error
                est_pose_quat = torch.cat((est_pose[:, :3],
                                           rot_utils.compute_quaternions_from_rotation_matrices(est_rot)), dim=1)
                posit_err, orient_err = utils.pose_err(est_pose_quat, gt_pose)

                # Collect statistics
                stats[i, 0] = posit_err.item()
                stats[i, 1] = orient_err.item()
                stats[i, 2] = (toc - tic) * 1000.0

        # Saving the models' evaluation results
        checkpoint_name = splitext(checkpoint_path.split('_')[-1])[0]
        batch_eval_results.append([checkpoint_name,
                                   checkpoint_path,
                                   np.nanmedian(stats[:, 0]),
                                   np.nanmedian(stats[:, 1])])

        logging.info("Median pose error: {:.3f}[m], {:.3f}[deg]".format(np.nanmedian(stats[:, 0]),
                                                                        np.nanmedian(stats[:, 1])))

    # Saving the results
    col_chk_pnt = 'checkpoint'
    col_model = 'Model'
    col_pos_err = 'Median Position Error [m]'
    col_orient_err = 'Median Orientation Error[deg]'
    batch_eval_results = pd.DataFrame(batch_eval_results, columns=[col_chk_pnt,
                                                                   col_model,
                                                                   col_pos_err,
                                                                   col_orient_err])

    results_file_prefix = cfg.inputs.models_path.split('/')[-1]
    batch_eval_results.to_csv(join(cfg.inputs.models_path, f'{results_file_prefix}_batch_eval.csv'))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=batch_eval_results[col_chk_pnt],
                             y=batch_eval_results[col_pos_err],
                             mode='lines+markers',
                             name=col_pos_err))
    fig.add_trace(go.Scatter(x=batch_eval_results[col_chk_pnt],
                             y=batch_eval_results[col_orient_err],
                             mode='lines+markers',
                             name=col_orient_err))
    fig.update_layout(
        title="Batch model evaluation: {}".format(cfg.inputs.model_name.capitalize()),
        xaxis_title="Model",
        yaxis_title="Position and Orientation Errors",
        legend_title="Error Type",
        font=dict(family="Courier New, monospace")
    )

    # Plotting and saving the figure
    plotly.offline.plot(fig, filename=join(cfg.inputs.models_path, f'{results_file_prefix}_batch_eval.html'))


if __name__ == "__main__":
    main()
