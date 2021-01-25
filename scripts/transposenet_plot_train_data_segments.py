import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import listdir
from os.path import join, isfile, exists, splitext
from tqdm import tqdm
import cv2
import plotly
import plotly.graph_objs as go
import cufflinks as cf
from plotly.offline import init_notebook_mode
import pickle
import matplotlib.cm as cm
from sklearn.cluster import KMeans

init_notebook_mode(connected=False)
cf.go_offline()


def verify_input_file(path):
    if not exists(path):
        print('File not found')
        exit(-1)
    else:
        print('Found file: {}'.format(path))


def get_dataset_images_names(paths):
    images_names = []
    for img_filename in tqdm(paths):
        images_names.append('/'.join(img_filename.split('/')[-2:]))
    return images_names


def gen_plotly_colors(num_of_colors, cmap_name='viridis', min_range=0.0, max_range=1.0):
    # Generate a color scale
    cmap = cm.get_cmap(cmap_name)

    # Select the color 75% of the way through the colorscale
    colors = []
    indices = np.linspace(min_range, max_range, num_of_colors)
    for color_idx in indices:
        rgba = cmap(color_idx)
        rgba = tuple(int((255 * x)) for x in rgba[0:3])
        colors.append('rgb' + str(rgba))
    return colors


def get_color_vect_by_label(colors, labels):
    color_vect = []
    for l in labels:
        color_vect.append(colors[l])
    return color_vect


# Setting input scene and path
# ============================
data_root_dir = '../datasets'
dataset_name = 'cambridge' #'7scenes'
scene = 'KingsCollege'

num_of_segments = 3
cluster_by = 'pose' #'netvlad_emb'

colors = gen_plotly_colors(num_of_segments, max_range=0.5)
fill_colors = gen_plotly_colors(num_of_segments, 'Greys', 0.2, 0.5)

if dataset_name == 'cambridge':
    dataset_root_dir = r'/media/ronf/data/datasets/cambridge'
else:
    dataset_root_dir = r'/media/ronf/data/datasets/7Scenes'

path = join(data_root_dir, dataset_name)

types = ['train', 'test']
scene_data = {'train': {}, 'test': {}}
data = []
kmeans = None

for type in types:
    # Retrieving input files
    files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    for f in files:
        if scene in f and type in f and splitext(f)[-1] == '.csv':
            input_file = f
    verify_input_file(input_file)

    # Reading training data
    # =====================
    scene_data[type]['data'] = pd.read_csv(input_file)

    # Plotting pose estimation error
    # =================================
    images_names = get_dataset_images_names(scene_data[type]['data']['img_path'])
    num_of_imgs = len(images_names)
    scene_data[type]['imgs'] = images_names

    if type == 'train':
        # Clustering the training data
        if cluster_by == 'pose':
            gt_pose = scene_data[type]['data'][['t1', 't2', 't3', 'q1', 'q2', 'q3', 'q4']].to_numpy()
            seg_input_data = gt_pose
        else:
            emb_file = join(data_root_dir, dataset_name,
                            'abs_{}_pose_sorted.csv_{}_train.csv_poselab_netvlad_embedding.pickle'.format(dataset_name,
                                                                                                          scene))
            with open(emb_file, "rb") as handle:
                emb_data = pickle.load(handle)
            seg_input_data = gt_pose

        kmeans = KMeans(n_clusters=num_of_segments, random_state=0).fit(seg_input_data)
        labels = kmeans.labels_
        for l in np.unique(labels):
            indices = l == labels
            data.append(go.Scatter(x=scene_data[type]['data']['t1'][indices].to_numpy(),
                                   y=scene_data[type]['data']['t2'][indices].to_numpy(),
                                   fill='toself', mode='none', fillcolor=fill_colors[l]))

        scatter_marker_outline = 0
        scatter_symbol = 'circle'

    if type == 'test':
        gt_pose = scene_data[type]['data'][['t1', 't2', 't3', 'q1', 'q2', 'q3', 'q4']].to_numpy()
        labels = kmeans.predict(gt_pose)

        scatter_marker_outline = 0.5
        scatter_symbol = 'cross'

    points_color = get_color_vect_by_label(colors, labels)
    data.append(go.Scatter(x=scene_data[type]['data']['t1'].to_numpy(),
                           y=scene_data[type]['data']['t2'].to_numpy(),
                           mode='markers',
                           marker=dict(symbol=scatter_symbol, color=points_color, line=dict(color='Black', width=scatter_marker_outline)),
                           name='{} Data'.format(type.title()),
                           text=list(map(lambda fn: f'File: ' + fn, images_names))
                           ))

    scene_name_with_label = ['{}{}'.format(scene, i) for i in labels]
    scene_data[type]['data']['scene'] = scene_name_with_label
    scene_data[type]['data'].to_csv(join(data_root_dir,
                                         input_file.replace('_{}.'.format(type),
                                                            '_MS_{}_{}_{}.'.format(num_of_segments, cluster_by, type))))


layout = go.Layout(title='Scene Data: <b>{}/{} - {} Segments</b>'.format(dataset_name.title(), scene, num_of_segments),
                   xaxis=dict(title='X Coordinate'),
                   yaxis=dict(title='Y Coordinate'))

save_path = r'../analysis/scene_train_data_plot_{}_{}.html'.format(dataset_name, scene)
plotly.offline.plot({'data':data, 'layout':layout}, filename=save_path, auto_open=True)