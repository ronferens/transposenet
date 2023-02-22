import torch
import torch.nn.functional as F
import torch.nn as nn
from util import rot_utils


class CameraPoseLoss(nn.Module):
    """
    A class to represent camera pose loss
    """

    def __init__(self, config):
        """
        :param config: (dict) configuration to determine behavior
        """
        super(CameraPoseLoss, self).__init__()
        self._learnable = config.get("learnable")
        self._s_x = torch.nn.Parameter(torch.Tensor([config.get("s_x")]), requires_grad=self._learnable)
        self._s_q = torch.nn.Parameter(torch.Tensor([config.get("s_q")]), requires_grad=self._learnable)
        self.norm = config.get("norm")

        self._mean_criterion = torch.nn.MSELoss(reduction='mean')

    def forward(self, est_loc, est_rot, gt_pose):
        """
        Forward pass
        :param est_pose: (torch.Tensor) batch of estimated poses, a Nx7 tensor
        :param gt_pose: (torch.Tensor) batch of ground_truth poses, a Nx7 tensor
        :return: camera pose loss
        """

        # Assigning the GT location and rotation
        gt_loc = gt_pose[:, :3]
        gt_rot = rot_utils.compute_rotation_matrix_from_quaternion(gt_pose[:, 3:])

        # Position loss
        l_x = torch.norm(gt_loc - est_loc, dim=1, p=self.norm).mean()

        # Orientation loss (normalized to unit norm)
        l_q = self._mean_criterion(est_rot, gt_rot)

        if self._learnable:
            return l_x * torch.exp(-self._s_x) + self._s_x + l_q * torch.exp(-self._s_q) + self._s_q
        else:
            return self._s_x * l_x + self._s_q * l_q
