import numpy as np
import matplotlib.pyplot as plt
import re

num_od_seg = 3
pose_err = []
rot_err = []

total_pose_err = []
total_rot_err = []

for idx in range(num_od_seg):
    filename = './kings_{}_test.log'.format(idx)
    file1 = open(filename, 'r')
    Lines = file1.readlines()

    p_err = []
    r_err = []
    for line in Lines:
        m = re.match('^.+ - Pose error: ([\d\.]+)\[m\], ([\d\.]+)\[deg\]', line)
        if m:
            p_err.append(float(m.group(1)))
            r_err.append(float(m.group(2)))

    pose_err.append(p_err)
    rot_err.append(r_err)

    total_pose_err += p_err
    total_rot_err += r_err

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].boxplot(pose_err, showfliers=False)
ax[0].set_xticklabels(['Seg-{}'.format(f) for f in range(num_od_seg)], fontsize=12)
ax[0].set_title('Translation Error per Segment')
ax[0].set_ylabel('Translation Error [m]')
ax[1].boxplot(rot_err, showfliers=False)
ax[1].set_xticklabels(['Seg-{}'.format(f) for f in range(num_od_seg)], fontsize=12)
ax[1].set_title('Orientation Error per Segment')
ax[1].set_ylabel('Orientation Error [deg]')
plt.show()

# Printing total pose error:
print('Median Pose error - {:.3f}[m], {:.3f}[deg]'.format(np.median(total_pose_err), np.median(total_rot_err)))