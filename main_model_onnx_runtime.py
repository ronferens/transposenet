import onnxruntime as ort
from tqdm import tqdm
import numpy as np

model_path = 'transposenet_pytorch.onnx'

providers = [
    # ('CPUExecutionProvider')
    # ('TensorrtExecutionProvider'),
    # ('CUDAExecutionProvider')
    # ('TensorrtExecutionProvider', {
    #     'device_id': 0,
    #     'trt_max_workspace_size': 2147483648,
    #     'trt_fp16_enable': True,
    # }),
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    })
]

sess_opt = ort.SessionOptions()
sess = ort.InferenceSession(model_path, sess_options=sess_opt, providers=providers)
# sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

# get the name of the first input of the model
input_name = sess.get_inputs()
label_name = sess.get_outputs()[0].name

inputs = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Time
from time import time

# Warmup
warmup_num_reps = 100
for i in tqdm(range(warmup_num_reps), desc='model warmup...'):
    sess.run([label_name], {input_name[0].name: inputs})

# Testing runtime
num_reps = 1000
tic = time()
for i in range(num_reps):
    sess.run([label_name], {input_name[0].name: inputs})
toc = time()
print('Runtime: {:.04f} ms'.format((1000.0 * (toc - tic) / num_reps)))