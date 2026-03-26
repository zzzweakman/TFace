import os
import sys
import json
import argparse
import numpy as np
import torch
from utils import perform_val_bin, get_val_data_from_bin
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
from frequency_utils import DEFAULT_LOW_FREQ_CHANNELS, expected_input_channels, frequency_tensor_from_images
from torchkit.backbone import get_model


def parse_args():
    parser = argparse.ArgumentParser(description='verfication tool')
    parser.add_argument('--ckpt_path', default=None, required=True, help='model_path')
    parser.add_argument('--backbone', default='MobileFaceNet', help='backbone type')
    parser.add_argument('--gpu_ids', default='0', help='gpu ids')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--data_root', default='', required=True, help='validation data root')
    parser.add_argument('--embedding_size', default=512, type=int, help='embedding_size')
    parser.add_argument('--preprocess_mode', default='rgb', choices=['rgb', 'high_freq', 'low_freq'])
    parser.add_argument('--freq_keep_channels', default=None, help='Optional comma-separated frequency channels to keep')
    parser.add_argument('--input_channels', default=None, type=int, help='Backbone input channel count override')
    parser.add_argument('--results_json', default=None, help='Optional output path for verification metrics in JSON')
    args = parser.parse_args()
    return args


def main():
    """ Perform evaluation on LFW, CFP-FP, AgeDB, CALFW, CPLFW datasets,
        each dataset consists of some positive and negative pair data.
    """
    args = parse_args()
    torch.manual_seed(1337)
    input_size = [112, 112]
    input_channels = args.input_channels
    if input_channels is None:
        input_channels = expected_input_channels(
            mode=args.preprocess_mode,
            keep_channels=args.freq_keep_channels,
            low_freq_channels=DEFAULT_LOW_FREQ_CHANNELS,
        )
    # load backbone
    backbone = get_model(args.backbone)(input_size, input_channel=input_channels)
    if not os.path.exists(args.ckpt_path):
        raise RuntimeError("%s not exists" % args.ckpt_path)
    backbone.load_state_dict(torch.load(args.ckpt_path))

    val_data_dir = args.data_root
    # load data
    lfw, cfp_fp, agedb_30, cplfw, calfw, \
        lfw_issame, cfp_fp_issame, agedb_30_issame, \
        cplfw_issame, calfw_issame = get_val_data_from_bin(val_data_dir)

    # backbone to gpu
    gpus = [int(x) for x in args.gpu_ids.rstrip().split(',')]
    visible_gpu_count = torch.cuda.device_count()
    print("CUDA_VISIBLE_DEVICES={}".format(os.environ.get("CUDA_VISIBLE_DEVICES", "")))
    print("Visible GPU count: {}".format(visible_gpu_count))
    print("Requested logical GPU ids: {}".format(gpus))
    if len(gpus) == 0:
        raise RuntimeError("No gpu id provided")
    if max(gpus) >= visible_gpu_count:
        raise RuntimeError("Requested gpu id {} exceeds visible gpu count {}".format(max(gpus), visible_gpu_count))
    if len(gpus) > 1:
        backbone = torch.nn.DataParallel(backbone, device_ids=gpus)
        backbone = backbone.cuda()
        print("Running with DataParallel on {} GPUs".format(len(gpus)))
    else:
        backbone = backbone.cuda()
        print("Running on single GPU")

    preprocess_fn = None
    if args.preprocess_mode != 'rgb':
        preprocess_fn = lambda inputs: frequency_tensor_from_images(
            inputs,
            mode=args.preprocess_mode,
            keep_channels=args.freq_keep_channels,
            low_freq_channels=DEFAULT_LOW_FREQ_CHANNELS,
        )

    print("Perform Evaluation on LFW, CFP_FP, AgeDB, CPLFW...")
    # LFW result
    accuracy_lfw, best_threshold_lfw = perform_val_bin(
        args.embedding_size,
        args.batch_size,
        backbone,
        lfw,
        lfw_issame,
        progress_desc='LFW',
        preprocess_fn=preprocess_fn)
    print("Evaluation: LFW Acc: {}, thresh: {}".format(accuracy_lfw, best_threshold_lfw))
    # CFP-FP result
    accuracy_cfp_fp, best_threshold_cfp_fp = perform_val_bin(
        args.embedding_size,
        args.batch_size,
        backbone,
        cfp_fp,
        cfp_fp_issame,
        progress_desc='CFP_FP',
        preprocess_fn=preprocess_fn)
    # AgeDB result
    print("Evaluation: CFP_FP Acc: {}, thresh: {}".format(accuracy_cfp_fp, best_threshold_cfp_fp))
    accuracy_agedb, best_threshold_agedb = perform_val_bin(
        args.embedding_size,
        args.batch_size,
        backbone,
        agedb_30,
        agedb_30_issame,
        progress_desc='AgeDB30',
        preprocess_fn=preprocess_fn)
    # CALFW result
    print("Evaluation: AgeDB Acc: {}, thresh: {}".format(accuracy_agedb, best_threshold_agedb))
    accuracy_calfw, best_threshold_calfw = perform_val_bin(
        args.embedding_size,
        args.batch_size,
        backbone,
        calfw,
        calfw_issame,
        progress_desc='CALFW',
        preprocess_fn=preprocess_fn)
    # CPLFW result
    print("Evaluation: CALFW Acc: {}, thresh: {}".format(accuracy_calfw, best_threshold_calfw))
    accuracy_cplfw, best_threshold_cplfw = perform_val_bin(
        args.embedding_size,
        args.batch_size,
        backbone,
        cplfw,
        cplfw_issame,
        progress_desc='CPLFW',
        preprocess_fn=preprocess_fn)
    print("Evaluation: CPLFW Acc: {}, thresh: {}".format(accuracy_cplfw, best_threshold_cplfw))

    if args.results_json:
        results = {
            'ckpt_path': args.ckpt_path,
            'backbone': args.backbone,
            'preprocess_mode': args.preprocess_mode,
            'input_channels': input_channels,
            'metrics': {
                'lfw': {'acc': float(accuracy_lfw), 'thresh': float(best_threshold_lfw)},
                'cfp_fp': {'acc': float(accuracy_cfp_fp), 'thresh': float(best_threshold_cfp_fp)},
                'agedb_30': {'acc': float(accuracy_agedb), 'thresh': float(best_threshold_agedb)},
                'calfw': {'acc': float(accuracy_calfw), 'thresh': float(best_threshold_calfw)},
                'cplfw': {'acc': float(accuracy_cplfw), 'thresh': float(best_threshold_cplfw)},
            },
        }
        with open(args.results_json, 'w') as f:
            json.dump(results, f, indent=2)
        print("Saved results JSON to {}".format(args.results_json))


if __name__ == "__main__":
    main()
