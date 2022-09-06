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


def sort_models_name(models_list):
    models_list = sorted(models_list, key=lambda x: (len(x), x))
    if '_final' in models_list[0] and len(models_list) > 1:
        models_list = models_list[1:] + models_list[:1]
    return models_list



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("model_name", help="name of model to create (e.g. posenet, transposenet")
    arg_parser.add_argument("backbone_path", help="path to backbone .pth - e.g. efficientnet")
    arg_parser.add_argument("dataset_path", help="path to the physical location of the dataset")
    arg_parser.add_argument("labels_file", help="path to a file mapping images to their poses")
    arg_parser.add_argument("models_path", help="path to folder containing the models to evaluate")
    arg_parser.add_argument("models_prefix", help="used for loading the clusters' centroids")
    arg_parser.add_argument("--models_to_evaluate", help="used for loading the clusters' centroids", type=str)
    arg_parser.add_argument("--results_file_suffix", help="a suffix for the saved .csv results file")
    arg_parser.add_argument("--verbose", help="print per-image results, for each experiment", default=False,
                            action='store_false')

    args = arg_parser.parse_args()
    utils.init_logger()

    # Record execution details
    logging.info("Running batch evaluation for - {}".format(args.model_name))
    logging.info("Using dataset: {}".format(args.dataset_path))
    logging.info("Using labels file: {}".format(args.labels_file))

    # Read configuration
    with open("config.json", "r") as read_file:
        config = json.load(read_file)
    model_params = config[args.model_name]
    general_params = config['general']
    config = {**model_params, **general_params}
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    np.random.seed(numpy_seed)
    device = torch.device(device_id)

    # Extract all checkpoints to evaluate
    all_model_files = [join(args.models_path, f) for f in listdir(args.models_path) if splitext(f)[-1] == '.pth' and
                       args.models_prefix in f]

    # Filter out the models
    if args.models_to_evaluate is not None:
        models_to_evaluate = [l for l in args.models_to_evaluate.split(',')]
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
        model = get_model(args.model_name, args.backbone_path, config).to(device)

        model.load_state_dict(torch.load(checkpoint_path, map_location=device_id))
        logging.info("Initializing from checkpoint: {}".format(checkpoint_path))

        # Set to eval mode
        model.eval()

        # Set the dataset and data loader
        transform = utils.test_transforms.get('baseline')
        dataset = CameraPoseDataset(args.dataset_path, args.labels_file, transform)
        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        stats = np.zeros((len(dataloader.dataset), 5))

        gt_cls = []
        preds_cls = []

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

                if args.verbose:
                    logging.info("Pose error: {:.3f}[m], {:.3f}[deg], inferred in {:.2f}[ms]".format(stats[i, 0],
                                                                                                     stats[i, 1],
                                                                                                     stats[i, 2]))

        # Record overall statistics
        if args.verbose:
            logging.info("Performance of {} on {}".format(checkpoint_path, args.labels_file))
            logging.info("\tMedian pose error: {:.3f}[m], {:.3f}[deg]".format(np.nanmedian(stats[:, 0]),
                                                                              np.nanmedian(stats[:, 1])))
            logging.info("\tMean inference time:{:.2f}[ms]".format(np.mean(stats[:, 2])))

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

    if args.results_file_suffix is not None:
        results_file_prefix = '{}_batch_eval_{}'.format(args.models_prefix, args.results_file_suffix)
    else:
        results_file_prefix = '{}_batch_eval_{}'.format(args.models_prefix, utils.get_stamp_from_log())
    batch_eval_results.to_csv(f'{results_file_prefix}.csv')

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
        title="Batch model evaluation: {}".format(args.model_name.capitalize()),
        xaxis_title="Model",
        yaxis_title="Position and Orientation Errors",
        legend_title="Error Type",
        font=dict(family="Courier New, monospace")
    )

    # Plotting and saving the figure
    plotly.offline.plot(fig, filename=f'{results_file_prefix}.html')
