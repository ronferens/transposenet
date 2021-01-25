""" main_embed_db
This script prints out an embedding of a database
"""
import commonprep
import argparse
import torch
from datautils.AbsPoseDataset import AbsPoseDataset
from datautils import augmentutils
from datautils import embedutils
from logutils import logutils
import logging
from algo.netvlad.NetVLAD import NetVLAD
from algo import algoutils
from datautils import iodataset

def embed(db_file, dataset_path, netvlad, device, transform, scene=None):
    """
    Embed the dataset using a NetVLAD model
    :param db_file: (str) the dataset file of the dataset to embed
    :param dataset_path: (str) the path to the physical location of the dataset
    :param netvlad: the netvlad model
    :param device: (torch.device) the device to use
    :param transform: (torchvision.transforms) the transformation to apply on the data
    :param scene: (str) a scene to embed, if None all dataset will be embedded
    :return: the NetVLAD embedding (N x 4096 tensor)
    """
    logging.info("Create the dataset and the loader")
    dbset = AbsPoseDataset(db_file, dataset_path, split=None, transform=transform, scene=scene)
    dbloader = torch.utils.data.DataLoader(dbset, **trainloader_params)
    logging.info("Start embedding {} images".format(len(dbset.paths)))
    db_embedding = embedutils.embed_dataset(dbloader, device, netvlad)
    logging.info("Embedding completed")
    return db_embedding

# Parameters
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("dataset_path", help="path to the physical location of the dataset")
    arg_parser.add_argument("db_file", help="the name of the database file (train) to embed")
    arg_parser.add_argument("embedder_path", help="a .pth file an embedder")
    arg_parser.add_argument("--img_size",
                            help="resizing size for image embedding")
    arg_parser.add_argument("--scene",
                            help="embed each scene separately, by default all scenes are embedded together",
                            action='store_true', default=False)
    args = arg_parser.parse_args()


    # Initialize log file
    logutils.init_logger()

    img_size = args.img_size
    if img_size is not None: # if the size is None we will do dynamic resizing
        img_size = int(img_size)
        batch_size = 4
    else:
        batch_size = 1

    logging.info("Initialize the data transformation and the data loader params")
    netvlad_transform = augmentutils.netvlad_transform(img_size)
    trainloader_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 4}

    # Create the model for image embedding and retrieval
    logging.info("Create the model for image embedding and retrieval")
    netvlad = NetVLAD()
    algoutils.load_state_dict(netvlad, args.embedder_path)
    embedder = netvlad
    embedder_transform = netvlad_transform
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if args.scene:
        scenes = iodataset.get_scenes(args.db_file)
        for s in scenes:
            logging.info("Embedding scene {}".format(s))
            db_embedding = embed(args.db_file, args.dataset_path, netvlad, device, netvlad_transform, scene=s)
            torch.save([db_embedding, embedder_transform], args.db_file + "_poselab_netvlad_embedding_{}.pth".format(s))
    else:
        db_embedding = embed(args.db_file, args.dataset_path, netvlad, device, netvlad_transform, scene=None)
        torch.save([db_embedding, embedder_transform], args.db_file + "_poselab_netvlad_embedding")









