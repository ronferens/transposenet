import argparse
import json
import logging
from util import utils
from models.pose_regressors import get_model
import torch.onnx


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("model_name",
                            help="name of model to create (e.g. posenet, transposenet")
    arg_parser.add_argument("backbone_path", help="path to backbone .pth - e.g. efficientnet")
    arg_parser.add_argument("--checkpoint_path",
                            help="path to a pre-trained model (should match the model indicated in model_name")

    args = arg_parser.parse_args()
    utils.init_logger()

    # Read configuration
    with open('config.json', "r") as read_file:
        config = json.load(read_file)
    model_params = config[args.model_name]
    general_params = config['general']
    config = {**model_params, **general_params}
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

    device_id = 'cuda:0'
    device = torch.device(device_id)

    # Create the model
    model = get_model(args.model_name, args.backbone_path, config).to(device)

    # Load the checkpoint if needed
    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device_id))
        logging.info("Initializing from checkpoint: {}".format(args.checkpoint_path))

    # ---------------------------
    # Export the model to ONNX
    # ---------------------------
    model.eval()
    input_names = ['input']
    output_names = ['output']

    BATCH_SIZE = 1
    dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224).to(device)

    # from time import time
    # warmup_num_reps = 100
    # for i in tqdm(range(warmup_num_reps), desc='model warmup...'):
    #     model(dummy_input)
    #
    # print('Now running model iteratively to calculate average runtime without acceleration')
    # num_reps = 1000
    # tic = time()
    # for i in range(num_reps):
    #     model(dummy_input)
    # toc = time()
    # print('Done')
    # print('Runtime: {:.04f} ms'.format((1000.0 * (toc - tic) / num_reps)))

    print('Exporting model to ONNX output file')
    onnx_outpout_model_path = "transposenet_pytorch.onnx"

    torch.onnx.export(model,
                      dummy_input,
                      onnx_outpout_model_path,
                      verbose=False,
                      input_names=input_names,
                      output_names=output_names,
                      do_constant_folding=True,
                      opset_version=11)
    # res = torch.onnx.export(model,
    #                         (dummy_input),
    #                         onnx_outpout_model_path,
    #                         input_names=input_names,
    #                         output_names=output_names,
    #                         do_constant_folding=True,
    #                         export_params=True,
    #                         verbose=True,
    #                         opset_version=11)

    print('Done')

    import onnx
    onnx_model = onnx.load(onnx_outpout_model_path)

    print('The model is:\n{}'.format(onnx_model))

    # Check the model
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print('The model is invalid: %s' % e)
    else:
        print('The model is valid!')

    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession(onnx_outpout_model_path, providers=['CPUExecutionProvider'])

    input_name = sess.get_inputs()
    inputs = np.random.randn(1, 3, 224, 224).astype(np.float32)

    outputs = sess.run(None, {input_name[0].name: inputs})
