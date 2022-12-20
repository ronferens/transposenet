"""
Entry point training and testing TransPoseNet
"""
import argparse
import torch
import numpy as np
import json
import logging
from util import utils
import time
from datasets.CameraPoseDataset import CameraPoseDataset
from models.pose_losses import CameraPoseLoss
import onnxruntime as ort
import numpy as np
from os.path import join


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("dataset_path", help="path to the physical location of the dataset")
    arg_parser.add_argument("labels_file", help="path to a file mapping images to their poses")
    arg_parser.add_argument("--checkpoint_path",
                            help="path to a pre-trained model (should match the model indicated in model_name")

    args = arg_parser.parse_args()
    utils.init_logger()

    # Record execution details
    logging.info("Using dataset: {}".format(args.dataset_path))
    logging.info("Using labels file: {}".format(args.labels_file))

    # Loading the ONNX deployed model
    model_path = 'transposenet_pytorch.onnx'

    providers = [
        # ('CPUExecutionProvider')
        # ('TensorrtExecutionProvider'),
        # ('CUDAExecutionProvider')
        # ('TensorrtExecutionProvider', {
        #     'device_id': 0,
        #     'trt_max_workspace_size': 2147483648,
        #     'trt_fp16_enable': True,
        # }),
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        })
    ]

    device = 'cuda:0'

    sess_opt = ort.SessionOptions()
    sess = ort.InferenceSession(model_path, sess_options=sess_opt, providers=providers)

    # get the name of the first input of the model
    input_name = sess.get_inputs()
    label_name = sess.get_outputs()[0].name

    # Set the dataset and data loader
    transform = utils.test_transforms.get('baseline')
    dataset = CameraPoseDataset(args.dataset_path, args.labels_file, transform)
    loader_params = {'batch_size': 1,
                     'shuffle': False,
                     'num_workers': 4}
    dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

    stats = np.zeros((len(dataloader.dataset), 3))

    with torch.no_grad():
        for i, minibatch in enumerate(dataloader, 0):
            for k, v in minibatch.items():
                minibatch[k] = v.to(device)

            gt_pose = minibatch.get('pose').data.cpu().numpy().astype(np.float32)
            inputs = minibatch.get('img').data.cpu().numpy().astype(np.float32)

            # Forward pass to predict the pose
            tic = time.time()
            est_pose = sess.run([label_name], {input_name[0].name: inputs})[0]
            toc = time.time()

            # Evaluate error
            posit_err, orient_err = utils.pose_err(torch.Tensor(est_pose), torch.Tensor(gt_pose))

            # Collect statistics
            stats[i, 0] = posit_err.item()
            stats[i, 1] = orient_err.item()
            stats[i, 2] = (toc - tic)*1000

            logging.info("Pose error: {:.3f}[m], {:.3f}[deg], inferred in {:.2f}[ms]".format(
                stats[i, 0],  stats[i, 1],  stats[i, 2]))

    # Record overall statistics
    logging.info("Performance of {} on {}".format(args.checkpoint_path, args.labels_file))
    logging.info("Median pose error: {:.3f}[m], {:.3f}[deg]".format(np.nanmedian(stats[:, 0]), np.nanmedian(stats[:, 1])))
    logging.info("Mean inference time:{:.2f}[ms]".format(np.mean(stats[:, 2])))