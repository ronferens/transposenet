"""
Entry point training and testing TransPoseNet
"""
import argparse
import pandas as pd
import torch
import numpy as np
import json
import logging
from util import utils
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


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg) -> None:
    utils.init_logger()

    # Record execution details
    logging.info("Start {} with {}".format(cfg.inputs.model_name, cfg.inputs.mode))
    if cfg.inputs.experiment is not None:
        logging.info("Experiment details: {}".format(cfg.inputs.experiment))
    logging.info("Using dataset: {}".format(cfg.inputs.dataset_path))
    logging.info("Using labels file: {}".format(cfg.inputs.labels_file))

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = cfg.general.device_id
    np.random.seed(numpy_seed)
    device = torch.device(device_id)

    # Extract all checkpoints to evaluate
    assert cfg.inputs.models_path is not None, 'You must specify the models_path'
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
        model = get_model(cfg.inputs.model_name, cfg.inputs.backbone_path, model_config).to(device)

        model.load_state_dict(torch.load(checkpoint_path, map_location=device_id))
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

        stats = np.zeros((len(dataloader.dataset), 5))

        with torch.no_grad():
            for i, minibatch in enumerate(dataloader, 0):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)

                gt_pose = minibatch.get('pose').to(dtype=torch.float32)

                # Forward pass to predict the pose
                tic = time.time()
                res = model(minibatch)
                toc = time.time()

                est_pose = res.get('pose')

                # Evaluate error
                posit_err, orient_err = utils.pose_err(est_pose, gt_pose)

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
