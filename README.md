<div align="center">
<h2>Paying Attention to Activation Maps in Camera Pose Regression</h2>

[**Yoli Shavit**](https://yolish.github.io/) · [**Ron Ferens**](https://ronferens.github.io/) · [**Yosi Keller**](https://yosikeller.github.io/)

Bar-Ilan University

<a href="https://arxiv.org/abs/2103.11477"><img src='https://img.shields.io/badge/arXiv-Transposenet-red' alt='Paper PDF'></a>
<a href='https://ronferens.github.io/transposenet/'><img src='https://img.shields.io/badge/Project_Page-Transposenet-green' alt='Project Page' target="_blank"></a>
</div>

This repository implement the *TransPoseNet* architecture described in our paper: [Paying Attention to Activation Maps in Camera Pose Regression](https://arxiv.org/pdf/2103.11477.pdf).

The figure below illustrates our proposed scheme. The input image is
		first encoded by a convolutional backbone. Two activation maps, at different resolutions, are transformed into sequential representations. The two activation sequences are analyzed by dual Transformer encoders, one per regression task. We depict the attention weights via
		heatmaps. Position is best estimated by corner-like image features,
		while orientation is estimated by edge-like features. Each Transformer encoder output is  used to regress the respective camera pose component (position x or orientation q).

![TransPoseNet Illustration](./img/transposenet.png)


---

### In a Nutshell

This code implements:

1. Training of a Transformer Encoder -based architecture for absolute pose regression 
2. Training of a PoseNet-like (CNN based) architecture (baseline)
3. Testing of the models implemented in 1-2

---

### Prerequisites

In order to run this repository you will need:

1. Python3 (tested with Python 3.7.7)
1. PyTorch deep learning framework (tested with version 1.0.0)
1. Use torch==1.4.0, torchvision==0.5.0
1. Download the [Cambridge Landmarks](http://mi.eng.cam.ac.uk/projects/relocalisation/#dataset) dataset and the [7Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) dataset

---

### Usage

The entry point for training and testing is the main.py script in the root directory

  For detailed explanation of the options run:
  ```
  python main.py -h
  ```
  For example, in order to train TransPoseNet on the ShopFacade scene from the CambridgeLandmarks dataset: 
  ```
python main.py transposenet train ./models/backbones/efficient-net-b0.pth <path to the CambridgeLandmarks dataset> ./datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_ShopFacade_train.csv
  ```
  Your checkpoints (.pth file saved based on the number you specify in the configuration file) and log file
  will be saved under an 'out' folder.
  
  In order to test your model, for example on the the ShopFacade scene:
  ```
python main.py transposenet test ./models/backbones/efficient-net-b0.pth <path to the CambridgeLandmarks dataset> ./datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_ShopFacade_test.csv --checkpoint_path <path to .pth>
  ```
  
  
  
  
  
