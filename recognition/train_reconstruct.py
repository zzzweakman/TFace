import logging
import os

import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from frequency_utils import (
    DEFAULT_LOW_FREQ_CHANNELS,
    expected_input_channels,
    frequency_tensor_from_images,
)
from torchkit.backbone import get_model
from torchkit.backbone.unet_reconstruction import count_parameters
from torchkit.data import MultiDataset, MultiImageListDataset
from torchkit.data import MultiDistributedSampler, ImageListDistributedSampler
from torchkit.task import BaseTask
from torchkit.util import AverageMeter, CkptLoader, Timer


class ReconstructionTask(BaseTask):
    """Train a reconstruction model from selected DCT frequency channels to RGB."""

    def __init__(self, cfg_file):
        super(ReconstructionTask, self).__init__(cfg_file)
        self.val_loader = None

    def preprocess_inputs(self, inputs):
        preprocess_mode = self.cfg.get("PREPROCESS_MODE", "high_freq")
        low_freq_channels = self.cfg.get("FREQ_LOW_CHANNELS", DEFAULT_LOW_FREQ_CHANNELS)
        keep_channels = self.cfg.get("FREQ_KEEP_CHANNELS", None)
        return frequency_tensor_from_images(
            inputs,
            mode=preprocess_mode,
            keep_channels=keep_channels,
            low_freq_channels=low_freq_channels,
            ratio=self.cfg.get("DCT_SAMPLING_RATIO", 8),
        )

    def make_model(self):
        model_name = self.cfg.get("MODEL_NAME", "UNetReconstruction")
        model_builder = get_model(model_name)
        input_channels = self.cfg["INPUT_CHANNELS"]
        output_channels = self.cfg.get("OUTPUT_CHANNELS", 3)
        base_channels = self.cfg.get("UNET_BASE_CHANNELS", 72)
        self.backbone = model_builder(
            input_channel=input_channels,
            output_channel=output_channels,
            base_channels=base_channels,
        )
        self.backbone.cuda()

        recon_params = count_parameters(self.backbone)
        trainable_params = count_parameters(self.backbone, trainable_only=True)
        logging.info(
            "{} Generated, params: {:.3f}M, trainable: {:.3f}M".format(
                model_name, recon_params / 1e6, trainable_params / 1e6
            )
        )

        try:
            ir50 = get_model("IR_50")(self.input_size, input_channel=3)
            logging.info("IR_50 reference params: {:.3f}M".format(count_parameters(ir50) / 1e6))
            del ir50
        except Exception as exc:
            logging.info("Skip IR_50 reference parameter count: {}".format(exc))

    def get_optimizer(self):
        optimizer_name = str(self.cfg.get("OPTIMIZER", "SGD")).lower()
        learning_rates = self.cfg["LRS"]
        init_lr = learning_rates[0]
        weight_decay = self.cfg["WEIGHT_DECAY"]
        if optimizer_name == "adam":
            return optim.Adam(self.backbone.parameters(), lr=init_lr, weight_decay=weight_decay)
        if optimizer_name == "adamw":
            return optim.AdamW(self.backbone.parameters(), lr=init_lr, weight_decay=weight_decay)
        return optim.SGD(
            self.backbone.parameters(),
            lr=init_lr,
            momentum=self.cfg.get("MOMENTUM", 0.9),
            weight_decay=weight_decay,
        )

    def compute_loss(self, outputs, targets):
        l1_loss = F.l1_loss(outputs, targets)
        mse_loss = F.mse_loss(outputs, targets)
        loss_name = str(self.cfg.get("RECON_LOSS", "l1_mse")).lower()
        if loss_name == "l1":
            total_loss = l1_loss
        elif loss_name == "mse":
            total_loss = mse_loss
        else:
            total_loss = l1_loss + self.cfg.get("MSE_WEIGHT", 0.1) * mse_loss
        return total_loss, l1_loss, mse_loss

    def update_log_and_summary(self, am_loss, am_l1, am_mse, am_psnr):
        scalars = {
            "train/loss": am_loss,
            "train/l1": am_l1,
            "train/mse": am_mse,
            "train/psnr": am_psnr,
        }
        self.update_summary({"scalars": scalars})
        log = {
            "loss": am_loss,
            "l1": am_l1,
            "mse": am_mse,
            "psnr": am_psnr,
        }
        self.update_log_buffer(log)

    def loop_step(self, epoch):
        self.backbone.train()
        am_loss = AverageMeter()
        am_l1 = AverageMeter()
        am_mse = AverageMeter()
        am_psnr = AverageMeter()
        t = Timer()

        for step, samples in enumerate(self.train_loader):
            self.call_hook("before_train_iter", step, epoch)

            targets = samples[0].cuda(non_blocking=True)
            inputs = self.preprocess_inputs(targets)

            if self.amp:
                with amp.autocast():
                    outputs = self.backbone(inputs)
                    loss, l1_loss, mse_loss = self.compute_loss(outputs, targets)
            else:
                outputs = self.backbone(inputs)
                loss, l1_loss, mse_loss = self.compute_loss(outputs, targets)

            self.backward_and_update(loss, [self.opt], self.scaler)

            batch_size = targets.size(0)
            psnr = self.psnr_from_mse(mse_loss.detach())
            am_loss.update(loss.detach().item(), batch_size)
            am_l1.update(l1_loss.detach().item(), batch_size)
            am_mse.update(mse_loss.detach().item(), batch_size)
            am_psnr.update(psnr.item(), batch_size)

            self.update_log_and_summary(am_loss, am_l1, am_mse, am_psnr)
            self.update_log_buffer({"time_cost": t.get_duration()})
            self.call_hook("after_train_iter", step, epoch)

    def make_validation_inputs(self):
        if not self.cfg.get("VALIDATE_EVERY_EPOCH", False):
            return

        val_datasets = self.cfg.get("VAL_DATASETS", None)
        if val_datasets is None:
            raise RuntimeError("VALIDATE_EVERY_EPOCH requires VAL_DATASETS")

        rgb_mean = self.cfg["RGB_MEAN"]
        rgb_std = self.cfg["RGB_STD"]
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=rgb_mean, std=rgb_std),
        ])

        val_names = [branch["name"] for branch in val_datasets]
        val_batch_sizes = [branch["batch_size"] for branch in val_datasets]
        dataset_mode = self.cfg.get("VAL_DATASET_MODE", self.cfg.get("DATASET_MODE", "auto")).lower()
        data_root = self.cfg.get("VAL_DATA_ROOT", self.cfg["DATA_ROOT"])
        index_root = self.cfg.get("VAL_INDEX_ROOT", self.cfg["INDEX_ROOT"])

        if dataset_mode == "image_list":
            ds = MultiImageListDataset(
                data_root,
                index_root,
                val_names,
                transform,
                extensions=self.cfg.get("IMAGE_EXTENSIONS", [".jpg", ".jpeg", ".png"]),
            )
        else:
            data_type = self.cfg.get("VAL_DATA_TYPE", self.cfg.get("DATA_TYPE", "auto")).lower()
            ds = MultiDataset(data_root, index_root, val_names, transform, data_type=data_type)
        ds.make_dataset(shard=True)

        if dataset_mode == "image_list":
            sampler = ImageListDistributedSampler(ds, val_batch_sizes)
        else:
            sampler = MultiDistributedSampler(ds, val_batch_sizes)
        self.val_loader = DataLoader(
            ds,
            sum(val_batch_sizes),
            shuffle=False,
            num_workers=self.cfg["NUM_WORKERS"],
            pin_memory=True,
            sampler=sampler,
            drop_last=False,
        )
        logging.info("Validation step_per_epoch = %d" % len(self.val_loader))

    def validate_epoch(self, epoch):
        if self.val_loader is None:
            return

        self.backbone.eval()
        am_loss = AverageMeter()
        am_l1 = AverageMeter()
        am_mse = AverageMeter()
        am_psnr = AverageMeter()
        max_batches = self.cfg.get("VAL_MAX_BATCHES", -1)

        with torch.no_grad():
            for step, samples in enumerate(self.val_loader):
                if max_batches > 0 and step >= max_batches:
                    break
                targets = samples[0].cuda(non_blocking=True)
                inputs = self.preprocess_inputs(targets)
                if self.amp:
                    with amp.autocast():
                        outputs = self.backbone(inputs)
                        loss, l1_loss, mse_loss = self.compute_loss(outputs, targets)
                else:
                    outputs = self.backbone(inputs)
                    loss, l1_loss, mse_loss = self.compute_loss(outputs, targets)

                batch_size = targets.size(0)
                psnr = self.psnr_from_mse(mse_loss.detach())
                am_loss.update(loss.detach().item(), batch_size)
                am_l1.update(l1_loss.detach().item(), batch_size)
                am_mse.update(mse_loss.detach().item(), batch_size)
                am_psnr.update(psnr.item(), batch_size)

        if self.rank == 0:
            logging.info(
                "Validation Epoch {} / {}, loss = [{:.6f}] l1 = [{:.6f}] mse = [{:.6f}] psnr = [{:.6f}]".format(
                    epoch + 1,
                    self.epoch_num,
                    am_loss.avg,
                    am_l1.avg,
                    am_mse.avg,
                    am_psnr.avg,
                )
            )
            self.write_validation_summary(epoch, am_loss, am_l1, am_mse, am_psnr)

        self.backbone.train()

    def write_validation_summary(self, epoch, am_loss, am_l1, am_mse, am_psnr):
        scalars = {
            "val/loss": am_loss.avg,
            "val/l1": am_l1.avg,
            "val/mse": am_mse.avg,
            "val/psnr": am_psnr.avg,
        }
        for hook in getattr(self, "_hooks", []):
            writer = getattr(hook, "writer", None)
            if writer is None:
                continue
            for key, value in scalars.items():
                writer.add_scalar(key, value, global_step=epoch + 1)

    def psnr_from_mse(self, mse):
        max_value = float(self.cfg.get("PSNR_MAX_VALUE", 2.0))
        eps = float(self.cfg.get("PSNR_EPS", 1e-8))
        return 20.0 * torch.log10(mse.new_tensor(max_value)) - 10.0 * torch.log10(mse + eps)

    def prepare(self):
        preprocess_mode = self.cfg.get("PREPROCESS_MODE", "high_freq")
        low_freq_channels = self.cfg.get("FREQ_LOW_CHANNELS", DEFAULT_LOW_FREQ_CHANNELS)
        keep_channels = self.cfg.get("FREQ_KEEP_CHANNELS", None)
        expected_channels = expected_input_channels(
            mode=preprocess_mode,
            keep_channels=keep_channels,
            low_freq_channels=low_freq_channels,
        )
        configured_channels = self.cfg.get("INPUT_CHANNELS", expected_channels)
        if configured_channels != expected_channels:
            raise RuntimeError(
                "INPUT_CHANNELS={} does not match preprocess setting {}, expected {}".format(
                    configured_channels,
                    preprocess_mode,
                    expected_channels,
                )
            )
        self.cfg["INPUT_CHANNELS"] = configured_channels
        self.make_inputs()
        self.make_validation_inputs()
        self.make_model()
        self.opt = self.get_optimizer()
        self.register_hooks()

    def load_pretrain_model(self):
        backbone_resume = self.cfg.get("BACKBONE_RESUME", "")
        if backbone_resume != "":
            CkptLoader.load_backbone(self.backbone, backbone_resume, self.local_rank)

        meta_resume = self.cfg.get("META_RESUME", "")
        if meta_resume != "":
            CkptLoader.load_meta(self.opt, self.scaler, self, meta_resume)

    def save_ckpt(self, epoch):
        model_root = self.cfg["MODEL_ROOT"]
        os.makedirs(model_root, exist_ok=True)
        if self.rank == 0:
            backbone_path = os.path.join(model_root, "Backbone_Epoch_%d_checkpoint.pth" % epoch)
            torch.save(self.backbone.module.state_dict(), backbone_path)

            meta_dict = {
                "EPOCH": epoch,
                "OPTIMIZER": self.opt.state_dict(),
            }
            if self.amp:
                meta_dict["AMP_SCALER"] = self.scaler.state_dict()
            meta_path = os.path.join(model_root, "META_Epoch_%d_checkpoint.pth" % epoch)
            torch.save(meta_dict, meta_path)
            logging.info("Save reconstruction checkpoint at epoch %d ..." % epoch)

    def train(self):
        self.prepare()
        self.call_hook("before_run")
        self.backbone = DistributedDataParallel(self.backbone, device_ids=[self.local_rank])
        for epoch in range(self.start_epoch, self.epoch_num):
            self.call_hook("before_train_epoch", epoch)
            self.loop_step(epoch)
            self.validate_epoch(epoch)
            self.call_hook("after_train_epoch", epoch)
        self.call_hook("after_run")


def main():
    task_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.environ.get(
        "TRAIN_CONFIG",
        os.path.join(task_dir, "train_casia_highfreq_reconstruct.yaml"),
    )
    task = ReconstructionTask(config_file)
    task.init_env()
    task.train()


if __name__ == "__main__":
    main()
