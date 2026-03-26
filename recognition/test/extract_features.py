import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torchvision.transforms as transforms

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
from frequency_utils import DEFAULT_LOW_FREQ_CHANNELS, expected_input_channels, frequency_tensor_from_images
from torchkit.backbone import get_model
from torchkit.util.utils import l2_norm
from datasets import IJBDataset


def parse_args():
    """ Parse input arguments.
    """
    parser = argparse.ArgumentParser(description='verfication tool')
    parser.add_argument('--ckpt_path', default=None, required=True, help='model_path')
    parser.add_argument('--backbone', default='MobileFaceNet', help='backbone type')
    parser.add_argument('--gpu_ids', default='0', help='gpu ids')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--data_root', default='', required=True, help='validation data root')
    parser.add_argument('--filename_list', default='', required=True, help='file_list')
    parser.add_argument('--embedding_size', default=512, help='embedding_size')
    parser.add_argument('--output_dir', default='', required=True, help='output dir')
    parser.add_argument('--preprocess_mode', default='rgb', choices=['rgb', 'high_freq', 'low_freq'])
    parser.add_argument('--freq_keep_channels', default=None, help='Optional comma-separated frequency channels to keep')
    parser.add_argument('--input_channels', default=None, type=int, help='Backbone input channel count override')
    args = parser.parse_args()
    return args


def de_preprocess(tensor):
    return tensor * 0.5 + 0.5


def hflip_batch(imgs_tensor):
    """ flip batch data
    """
    hflip = transforms.Compose([
            de_preprocess,
            transforms.ToPILImage(),
            transforms.functional.hflip,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)

    return hfliped_imgs


def extract_features(data_loader, backbone, preprocess_fn=None):
    """ extract features from origin img and flipped,
        concated into single feature
    """
    embeddings = []
    flip_embeddings = []
    faceness_scores = []
    backbone.eval()
    print("Number of Test Image: {}".format(len(data_loader)))
    with torch.no_grad():
        for batch, label in tqdm(iter(data_loader)):
            fliped = hflip_batch(batch)
            batch = batch.cuda()
            fliped = fliped.cuda()
            if preprocess_fn is not None:
                batch = preprocess_fn(batch)
                fliped = preprocess_fn(fliped)
            feature_1 = backbone(batch)
            feature_2 = backbone(fliped)
            feature_1 = l2_norm(feature_1)
            feature_2 = l2_norm(feature_2)
            embedding = feature_1.cpu().numpy()
            flip_embedding = feature_2.cpu().numpy()
            embeddings.append(embedding)
            flip_embeddings.append(flip_embedding)
            faceness_scores.append(label)
        embeddings = np.concatenate(embeddings, axis=0)
        flip_embeddings = np.concatenate(flip_embeddings, axis=0)
        embeddings = np.concatenate([embeddings, flip_embeddings], axis=1)
        faceness_scores = np.concatenate(faceness_scores, axis=0)
        print('feature size', embeddings.shape)
        print('face score size', faceness_scores.shape)
        return embeddings, faceness_scores


def main():
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
    backbone = get_model(args.backbone)(input_size, input_channel=input_channels)
    if not os.path.exists(args.ckpt_path):
        raise RuntimeError("%s not exists" % args.ckpt_path)

    # load ckpt
    backbone.load_state_dict(torch.load(args.ckpt_path))

    gpus = [int(x) for x in args.gpu_ids.rstrip().split(',')]
    if len(gpus) > 1:
        backbone = torch.nn.DataParallel(backbone, device_ids=gpus)
        backbone = backbone.cuda()
    else:
        backbone = backbone.cuda()

    preprocess_fn = None
    if args.preprocess_mode != 'rgb':
        preprocess_fn = lambda inputs: frequency_tensor_from_images(
            inputs,
            mode=args.preprocess_mode,
            keep_channels=args.freq_keep_channels,
            low_freq_channels=DEFAULT_LOW_FREQ_CHANNELS,
        )

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])])
    dataset_test = IJBDataset(args.data_root, args.filename_list, test_transform)
    test_loader = torch.utils.data.DataLoader(
                dataset_test, batch_size=args.batch_size, pin_memory=True,
                num_workers=5, shuffle=False)
    # extract features and scores
    embeddings, faceness_scores = extract_features(test_loader, backbone, preprocess_fn=preprocess_fn)
    output_dir = args.output_dir
    # save features and scores
    feature_outputname = os.path.join(output_dir, 'feature.npy')
    faceness_scores_outputname = os.path.join(output_dir, 'faceness_scores.npy')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(feature_outputname, embeddings)
    np.save(faceness_scores_outputname, faceness_scores)
    print("extract feature done")


if __name__ == "__main__":
    main()
