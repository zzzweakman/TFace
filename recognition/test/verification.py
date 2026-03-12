import os
import sys
import argparse
import numpy as np
import torch
from utils import perform_val_bin, get_val_data_from_bin
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
from torchkit.backbone import get_model


def parse_args():
    parser = argparse.ArgumentParser(description='verfication tool')
    parser.add_argument('--ckpt_path', default=None, required=True, help='model_path')
    parser.add_argument('--backbone', default='MobileFaceNet', help='backbone type')
    parser.add_argument('--gpu_ids', default='0', help='gpu ids')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--data_root', default='', required=True, help='validation data root')
    parser.add_argument('--embedding_size', default=512, type=int, help='embedding_size')
    args = parser.parse_args()
    return args


def main():
    """ Perform evaluation on LFW, CFP-FP, AgeDB, CALFW, CPLFW datasets,
        each dataset consists of some positive and negative pair data.
    """
    args = parse_args()
    torch.manual_seed(1337)
    input_size = [112, 112]
    # load backbone
    backbone = get_model(args.backbone)(input_size)
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

    print("Perform Evaluation on LFW, CFP_FP, AgeDB, CPLFW...")
    # LFW result
    accuracy_lfw, best_threshold_lfw = perform_val_bin(
        args.embedding_size,
        args.batch_size,
        backbone,
        lfw,
        lfw_issame,
        progress_desc='LFW')
    print("Evaluation: LFW Acc: {}, thresh: {}".format(accuracy_lfw, best_threshold_lfw))
    # CFP-FP result
    accuracy_cfp_fp, best_threshold_cfp_fp = perform_val_bin(
        args.embedding_size,
        args.batch_size,
        backbone,
        cfp_fp,
        cfp_fp_issame,
        progress_desc='CFP_FP')
    # AgeDB result
    print("Evaluation: CFP_FP Acc: {}, thresh: {}".format(accuracy_cfp_fp, best_threshold_cfp_fp))
    accuracy_agedb, best_threshold_agedb = perform_val_bin(
        args.embedding_size,
        args.batch_size,
        backbone,
        agedb_30,
        agedb_30_issame,
        progress_desc='AgeDB30')
    # CALFW result
    print("Evaluation: AgeDB Acc: {}, thresh: {}".format(accuracy_agedb, best_threshold_agedb))
    accuracy_calfw, best_threshold_calfw = perform_val_bin(
        args.embedding_size,
        args.batch_size,
        backbone,
        calfw,
        calfw_issame,
        progress_desc='CALFW')
    # CPLFW result
    print("Evaluation: CALFW Acc: {}, thresh: {}".format(accuracy_calfw, best_threshold_calfw))
    accuracy_cplfw, best_threshold_cplfw = perform_val_bin(
        args.embedding_size,
        args.batch_size,
        backbone,
        cplfw,
        cplfw_issame,
        progress_desc='CPLFW')
    print("Evaluation: CPLFW Acc: {}, thresh: {}".format(accuracy_cplfw, best_threshold_cplfw))


if __name__ == "__main__":
    main()
