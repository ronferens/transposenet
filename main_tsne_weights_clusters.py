import argparse
import random

import cv2
import numpy as np
from PIL import Image, ImageOps
from os.path import join, exists
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tqdm import tqdm
import pandas as pd
from sklearn.decomposition import PCA


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


def draw_rect_by_color(img, color, thickness):
    return ImageOps.expand(img, border=thickness, fill=color)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('dataset_name', help='The dataset name', type=str)
    arg_parser.add_argument('scene_data_path', help='The path to the scene\'s images folder', type=str)
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

    for cluster_type in ['location', 'weights']:
        data = []

        images_names = get_dataset_images_names(scene_data['img_path'].values)
        images_seq = scene_data['seq'].values
        num_of_imgs = len(images_names)

        # Clustering the training data and set initial labels
        if cluster_type == 'location':
            data_to_cluster = scene_data[['t1', 't2', 't3']].to_numpy()
            num_clusters = args.num_clusters
            n_components = 2

            data_to_cluster = data_to_cluster / np.linalg.norm(data_to_cluster, axis=0)
            labels = get_labels_for_ordinal_classification(num_clusters, data_to_cluster)

        elif cluster_type == 'weights':
            weights_data = pd.read_csv(args.weights_path)
            data_to_cluster = weights_data.iloc[:, :387].to_numpy()
            num_clusters = args.num_clusters
            n_components = 16

            data_to_cluster = data_to_cluster / np.linalg.norm(data_to_cluster, axis=0)
        else:
            raise ValueError

        data = []
        for idx, filename in tqdm(enumerate(images_names)):
            data.append([data_to_cluster[idx, :], join(args.scene_data_path, filename)])

        features, images  = zip(*data)
        features = np.array(features)
        pca = PCA(n_components=n_components)
        pca.fit(features)
        pca_features = pca.transform(features)
        num_images_to_plot = len(images)

        if len(images) > num_images_to_plot:
            sort_order = sorted(random.sample(range(len(images)), num_images_to_plot))
            images = [images[i] for i in sort_order]
            pca_features = [pca_features[i] for i in sort_order]
        X = np.array(pca_features)
        tsne = TSNE(n_components=2, learning_rate=350, perplexity=30, angle=0.2, verbose=2).fit_transform(X)

        tx, ty = tsne[:,0], tsne[:,1]
        tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

        import matplotlib.pyplot as plt

        width = 4000
        height = 3000
        max_dim = 250
        colors = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0), (0, 0, 255)]

        idx = 0
        full_image = Image.new('RGBA', (width, height))
        for img, x, y in zip(images, tx, ty):
            label = labels[idx]
            idx += 1

            tile = Image.open(img)
            rs = max(1, tile.width/max_dim, tile.height/max_dim)
            tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
            tile = draw_rect_by_color(tile, colors[label], 5)
            full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

        plt.figure(figsize = (16, 12))
        plt.imshow(full_image)
        plt.tick_params(labelleft=False, labelbottom=False, bottom=False, left=False)
        plt.show()