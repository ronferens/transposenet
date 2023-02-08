import numpy as np
import pandas as pd
from os.path import join, splitext, exists
from tqdm import tqdm
import plotly
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode
from sklearn.cluster import KMeans
import argparse
from sklearn.decomposition import PCA

init_notebook_mode(connected=False)


def verify_input_file(path):
    if not exists(path):
        print('File not found')
        exit(-1)
    else:
        print('Found file: {}'.format(path))


def get_dataset_images_names(paths):
    names = []
    for img_filename in tqdm(paths, desc='loading dataset images'):
        names.append('/'.join(img_filename.split('/')[-2:]))
    return names


def get_cluster_centroids(data, num_of_segments, labels):
    # Finding the centroids of the 1D clusters
    centroids = np.zeros((num_of_segments, data.shape[1]))
    for i in range(num_of_segments):
        indices = i == labels
        cent = np.mean(data[indices], axis=0)
        centroids[i, :] = cent
    return centroids


def get_labels_for_ordinal_classification(num_of_segments, data):
    # Clustering the input data
    kmeans = KMeans(n_clusters=num_of_segments, random_state=0).fit(data)

    # Reprojecting the clustered data into 1D using LDA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=1)
    data_1d = lda.fit(data, kmeans.labels_).transform(data)

    # Finding the centroids of the 1D clusters
    centroids_1d = np.zeros(num_of_segments)
    for i in range(num_of_segments):
        indices = i == kmeans.labels_
        cent = np.mean(data_1d[indices])
        centroids_1d[i] = cent

    # Setting the order of the 1D centroids
    org_labels_order = np.argsort(centroids_1d.reshape(1, -1))[0]

    # Assigning the new labels
    new_labels_order = np.arange(num_of_segments)
    new_labels = np.zeros_like(kmeans.labels_)
    for new_l, l in enumerate(org_labels_order):
        indices = l == kmeans.labels_
        new_labels[indices] = new_labels_order[new_l]

    return new_labels


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('dataset_name', help='The dataset name', type=str)
    arg_parser.add_argument('dataset_path', help='The path to the dataset .csv file', type=str)
    arg_parser.add_argument('scene', help='The name of the scene to cluster', type=str)
    arg_parser.add_argument('num_clusters', help='Number of clusters in the dataset', type=int)
    arg_parser.add_argument('--weights_path', type=str)
    arg_parser.add_argument('--viz', help='Indicates whether to visualize the positional clustering',
                            action='store_true', default=False)
    args = arg_parser.parse_args()

    # Setting input scene and path
    # ============================
    input_file = args.dataset_path
    scene = args.scene

    # Reading the train data
    scene_data = pd.read_csv(input_file)

    for cluster_type in ['location', 'weights']: # ['location', 'orientation', 'pose']:
        data = []

        images_names = get_dataset_images_names(scene_data['img_path'].values)
        images_seq = scene_data['seq'].values
        num_of_imgs = len(images_names)

        # Clustering the training data and set initial labels
        if cluster_type == 'location':
            data_to_cluster = scene_data[['t1', 't2', 't3']].to_numpy()
            num_clusters = args.num_clusters
        elif cluster_type == 'orientation':
            data_to_cluster = scene_data[['q1', 'q2', 'q3', 'q4']].to_numpy()
            num_clusters = args.num_clusters
        elif cluster_type == 'pose':
            data_to_cluster = scene_data[['t1', 't2', 't3', 'q1', 'q2', 'q3', 'q4']].to_numpy()
            num_clusters = args.num_clusters
        elif cluster_type == 'weights':
            weights_data = pd.read_csv(args.weights_path)
            data_to_cluster = weights_data.iloc[:, :387].to_numpy()
            num_clusters = args.num_clusters
        else:
            raise ValueError

        data_to_cluster = data_to_cluster / np.linalg.norm(data_to_cluster, axis=0)

        labels = get_labels_for_ordinal_classification(num_clusters, data_to_cluster)
        cluster_centroids = get_cluster_centroids(data_to_cluster, num_clusters, labels)
        centroids = np.zeros(data_to_cluster.shape)

        # Visualizing only for positional clusters (using X/Y coordinates)
        for label in np.unique(labels):
            indices = label == labels
            data.append(go.Scatter(x=scene_data['t1'][indices].to_numpy(),
                                   y=scene_data['t2'][indices].to_numpy(),
                                   mode='markers',
                                   marker=dict(size=25, line=dict(color='DarkSlateGrey', width=5)),
                                   name='cluster #{}'.format(label),
                                   text=list(map(lambda fn: f'File: ' + fn, np.array(images_names)[indices].tolist()))))
            centroids[indices, :] = cluster_centroids[label]

        # Adding the labels to the dataset data
        scene_data['class_{}'.format(cluster_type)] = labels
        for i in range(centroids.shape[1]):
            scene_data['cent_{}_{}'.format(cluster_type, (i + 1))] = centroids[:, i]

        if args.viz:
            layout = go.Layout(title='Scene Data: <b>{}/{} - {} Segments - {}</b>'.format(args.dataset_name.title(),
                                                                                          scene,
                                                                                          num_clusters,
                                                                                          cluster_type.title()),
                               xaxis=dict(showticklabels=False, title=''),
                               yaxis=dict(showticklabels=False, title=''))

            save_path = r'{}_{}_{}_segments_{}.html'.format(args.dataset_name, scene, num_clusters, cluster_type)
            plotly.offline.plot({'data': data, 'layout': layout}, filename=save_path, auto_open=True)

        # Saving the dataset data
        output_file_path = splitext(input_file)[0] + '_{}_classes_{}'.format(args.num_clusters, cluster_type) + \
                           splitext(input_file)[1]
        scene_data.to_csv(output_file_path)
