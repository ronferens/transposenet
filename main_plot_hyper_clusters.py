from sklearn.manifold import TSNE
import pandas as pd


def main():
    # Loading the embedding and the corresponding labels
    data = pd.read_csv('XXX')
    x = data[:, ].values
    label = data[:, ].values

    # Fitting an t-SNE
    tsne = TSNE()
    x_embds = tsne.fit_transform(x)


if __name__ == '__main__':
    main()