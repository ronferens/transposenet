import argparse
import cv2
import numpy as np
from os.path import join, exists
from os import mkdir
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import pickle


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


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('dataset_name', help='The dataset name', type=str)
    arg_parser.add_argument('scene_data_path', help='The path to the scene\'s images folder', type=str)
    arg_parser.add_argument('dataset_path', help='The path to the dataset .csv file', type=str)
    arg_parser.add_argument('scene', help='The name of the scene to cluster', type=str)
    arg_parser.add_argument('output_path', type=str)
    arg_parser.add_argument('weights_path', type=str)
    arg_parser.add_argument('--viz', help='Indicates whether to visualize the positional clustering',
                            action='store_true', default=False)
    args = arg_parser.parse_args()

    # Setting input scene and path
    # ============================
    input_file = args.dataset_path
    scene = args.scene

    # Reading the train data
    scene_data = pd.read_csv(input_file)

    if not exists(args.output_path):
        mkdir(args.output_path)

    # weights_data = pd.read_csv(args.weights_path)
    file = open('weights_in', 'rb')
    weights_data = pickle.load(file)
    file.close()

    weights_dim = 65792
    weights_2d_dim = 256
    weights_data = weights_data.iloc[:, :weights_dim].to_numpy()

    images_names = get_dataset_images_names(scene_data['img_path'].values)
    images_seq = scene_data['seq'].values
    num_of_imgs = len(images_names)

    for idx, filename in tqdm(enumerate(images_names)):
        img = cv2.imread(join(args.scene_data_path, filename))
        cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_CUBIC)

        act_img_weights = weights_data[idx, :].reshape(-1, weights_2d_dim, (weights_2d_dim + 1))[:, :, :weights_2d_dim]
        act_img_weights = np.squeeze(act_img_weights[:, :, :-1])
        act_img_weights[act_img_weights < (0.25 * np.median(act_img_weights))] = 0
        act_img_weights[act_img_weights > (0.75 * np.median(act_img_weights))] = 0
        act_img = np.mean(act_img_weights, axis=1)
        act_img = 255 * (act_img - act_img.min()) / (act_img.max() - act_img.min())
        act_img = np.uint8(act_img)
        act_img = act_img.reshape(16, 16)
        disp_act_img = cv2.resize(act_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

        plt.figure()
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.imshow(disp_act_img, alpha=0.3, cmap='jet')
        plt.axis('off')
        if args.viz:
            plt.show()
        else:
            plt.savefig(join(args.output_path, f'{scene}_img_{idx}.png'))
