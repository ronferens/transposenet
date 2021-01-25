import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import listdir
from os.path import join, isfile, exists
from tqdm import tqdm
import cv2
import plotly
import plotly.graph_objs as go

import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=False)
cf.go_offline()


def verify_input_file(path):
    if not exists(path):
        print('File not found')
        exit(-1)
    else:
        print('Found file: {}'.format(path))


def pose_error(est_pose, gt_pose):
    # Calculating the traslation error
    t_err = np.linalg.norm(est_pose[:, :3] - gt_pose[:, :3], axis=1)

    # Calculating the orientation error
    temp = abs(np.sum(gt_pose[:, 3:] * est_pose[:, 3:], axis=1))
    valid = temp <= 1  # Stability problem when values > 1 inside the argument - accures when comparing same values
    temp[~valid] = 1
    rot_err_rad = 2 * np.arccos(temp)
    rot_err_deg = np.rad2deg(rot_err_rad)

    return t_err, rot_err_deg


def norm_error(err):
    return (err - np.min(err)) / (np.max(err) - np.min(err))


def load_dataset_images(root, paths):
    images_data = []
    images_names = []
    for img_filename in tqdm(paths):
        # with open(join(root, img_filename), "rb") as f:
        #     b = f.read()
        # images_data.append(b)
        images_names.append('/'.join(img_filename.split('/')[-2:]))
    return images_data, images_names


# Setting input scene and path
# ============================
data_root_dir = r'\\magneto\data\Users\Ron\TransPoseNet\datasets'
dataset_name = '7scenes'#'cambridge'
type = 'test'
scene = 'chess'

if dataset_name == 'cambridge':
    dataset_root_dir = r'M:\Datasets\CAMBRIDGE_dataset'
else:
    dataset_root_dir = r'M:\Datasets\7Scenes'

path = join(data_root_dir, dataset_name)

# Retrieving input files
files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
for f in files:
    if scene in f and type in f:
        input_file = f
verify_input_file(input_file)

# Setting path to results data
# ============================
results_root_dir = r'\\magneto\data\Users\Ron\TransPoseNet\bechmark_results'
res_file = join(results_root_dir, 'transposenet_{}_{}.csv'.format(dataset_name, scene))
verify_input_file(res_file)

# Calculating pose estimation error
# =================================
scene_data = pd.read_csv(input_file)
res_data = pd.read_csv(res_file)

gt_pose = scene_data[['t1', 't2', 't3', 'q1', 'q2', 'q3', 'q4']].to_numpy()
est_pose = res_data[['t1', 't2', 't3', 'q1', 'q2', 'q3', 'q4']].to_numpy()
pos_err, rot_err = pose_error(est_pose, gt_pose)

th = 0.75 * np.max(pos_err)
pos_err[pos_err > th] = th
th = 0.75 * np.max(rot_err)
rot_err[rot_err > th] = th
norm_pos_err = norm_error(pos_err)
norm_rot_err = norm_error(rot_err)
est_err = (norm_pos_err + norm_rot_err) / 2.0

# Plotting pose estimation error
# =================================
images_data, images_names = load_dataset_images(dataset_root_dir, scene_data['img_path'])

data = [go.Scatter(x=scene_data['t1'].to_numpy(),
                   y=scene_data['t2'].to_numpy(),
                   mode='markers+lines',
                   name='Pose Estimation Error',
                   text=list(map(lambda fn, pe, re: f'File: ' + fn +
                                                '<br>Pose Error: ' + str(pe) +
                                                '<br>Rot Error: ' + str(re),
                                 images_names,
                                 np.round(pos_err, 3),
                                 np.round(rot_err, 3))),
                   marker=dict(
                       color=est_err, size=16,
                       colorbar=dict(thickness=5, tickvals=[0, 1], ticktext=['Low', 'High'], outlinewidth=0))
                   )]

layout = go.Layout(title='Pose Estimation Error - Scene: <b>{}/{}</b>'.format(dataset_name.title(), scene),
                   xaxis=dict(title='X Coordinate'),
                   yaxis=dict(title='Y Coordinate'))
# plotly.offline.plot({'data': data, 'layout': layout}, auto_open=True)

save_path = r'\\magneto\data\Users\Ron\TransPoseNet\pose_error_plot_{}_{}.html'.format(dataset_name, scene)
plotly.offline.plot({'data':data, 'layout':layout}, filename=save_path, auto_open=True)