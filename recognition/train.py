import os
import torch
import torch.cuda.amp as amp
from torch.nn.parallel import DistributedDataParallel

from frequency_utils import (
    DEFAULT_LOW_FREQ_CHANNELS,
    expected_input_channels,
    frequency_tensor_from_images,
)
from torchkit.backbone.channel_gate import FrequencyChannelGate
from torchkit.util import AverageMeter, Timer
from torchkit.util import accuracy_dist
from torchkit.util import AllGather

from torchkit.loss import get_loss
from torchkit.task import BaseTask


class TrainTask(BaseTask):
    """ TrainTask in distfc mode, which means classifier shards into multi workers
    """
    def __init__(self, cfg_file):
        super(TrainTask, self).__init__(cfg_file)
        
    def update_log_and_summary(self, am_loss, am_top1, am_top5):
        scalars = {
            'train/loss': am_loss,
            'train/top1': am_top1,
            'train/top5': am_top5,
        }
        self.update_summary({'scalars': scalars})
        log = {
            'loss': am_loss,
            'prec@1': am_top1,
            'prec@5': am_top5,
        }
        self.update_log_buffer(log)

    def maybe_add_channel_gate(self):
        if not self.cfg.get('FREQ_CHANNEL_GATE', False):
            return
        init_value = self.cfg.get('FREQ_CHANNEL_GATE_INIT', 0.9)
        use_sigmoid = self.cfg.get('FREQ_CHANNEL_GATE_SIGMOID', True)
        self.backbone = FrequencyChannelGate(
            self.backbone,
            num_channels=self.cfg['INPUT_CHANNELS'],
            init_value=init_value,
            use_sigmoid=use_sigmoid,
        )

    def get_channel_gate(self):
        module = self.backbone.module if hasattr(self.backbone, 'module') else self.backbone
        if isinstance(module, FrequencyChannelGate):
            return module
        return None

    def channel_gate_regularization(self):
        gate = self.get_channel_gate()
        weight = self.cfg.get('FREQ_CHANNEL_GATE_L1_WEIGHT', 0.0)
        if gate is None or weight <= 0:
            return None
        return gate.gate_l1() * weight

    def preprocess_inputs(self, inputs):
        preprocess_mode = self.cfg.get('PREPROCESS_MODE', 'rgb')
        if str(preprocess_mode).lower() == 'rgb':
            return inputs
        low_freq_channels = self.cfg.get('FREQ_LOW_CHANNELS', DEFAULT_LOW_FREQ_CHANNELS)
        keep_channels = self.cfg.get('FREQ_KEEP_CHANNELS', None)
        return frequency_tensor_from_images(
            inputs,
            mode=preprocess_mode,
            keep_channels=keep_channels,
            low_freq_channels=low_freq_channels,
            ratio=self.cfg.get('DCT_SAMPLING_RATIO', 8),
        )
        
    def loop_step(self, epoch):
        """
        load_data
            |
        extract feature
            |
        optimizer step
            |
        print log and write summary
        """
        backbone, heads = self.backbone, list(self.heads.values())
        backbone.train()  # set to training mode
        for head in heads:
            head.train()

        batch_sizes = self.batch_sizes
        am_losses = [AverageMeter() for _ in batch_sizes]
        am_top1s = [AverageMeter() for _ in batch_sizes]
        am_top5s = [AverageMeter() for _ in batch_sizes]
        t = Timer()
        for step, samples in enumerate(self.train_loader):
            # call hook function before_train_iter
            self.call_hook("before_train_iter", step, epoch)
            backbone_opt, head_opts = self.opt['backbone'], list(self.opt['heads'].values())

            inputs = samples[0].cuda(non_blocking=True)
            labels = samples[1].cuda(non_blocking=True)
            inputs = self.preprocess_inputs(inputs)

            all_features, all_labels = self.backbone_forward(backbone, inputs, labels, batch_sizes)

            losses = []
            for i in range(len(batch_sizes)):
                # PartialFC need update optimizer state in training process
                if self.pfc: 
                    outputs, labels, original_outputs = self.partialfc_head_forward(heads[i], all_features[i],
                                                                                    all_labels[i], head_opts[i])
                else:
                    outputs, labels, original_outputs = self.general_head_forward(heads[i], all_features[i],
                                                                                  all_labels[i])

                loss = self.loss(outputs, labels) * self.branch_weights[i]
                losses.append(loss)
                precs = accuracy_dist(self.cfg,
                                             original_outputs.data,
                                             all_labels[i],
                                             self.class_shards[i],
                                             topk=(1, 5))
                prec1, prec5 = precs
                am_losses[i].update(loss.data.item(), all_features[i].size(0))
                am_top1s[i].update(prec1.data.item(), all_features[i].size(0))
                am_top5s[i].update(prec5.data.item(), all_features[i].size(0))

            # update summary and log_buffer
            self.update_log_and_summary(am_losses, am_top1s, am_top5s)

            # compute loss
            total_loss = sum(losses)
            gate_reg = self.channel_gate_regularization()
            if gate_reg is not None:
                total_loss = total_loss + gate_reg
                self.update_log_buffer({'gate_l1': gate_reg.detach().item()})
            # compute gradient and do SGD
            total_opts = [backbone_opt] + head_opts
            self.backward_and_update(total_loss, total_opts, self.scaler)

            # PartialFC need update weight and weight_norm manually
            if self.pfc:
                for head in heads:
                    head.update()

            cost = t.get_duration()
            self.update_log_buffer({'time_cost': cost})

            # call hook function after_train_iter
            self.call_hook("after_train_iter", step, epoch)

    def prepare(self):
        """ common prepare task for training
        """
        preprocess_mode = self.cfg.get('PREPROCESS_MODE', 'rgb')
        low_freq_channels = self.cfg.get('FREQ_LOW_CHANNELS', DEFAULT_LOW_FREQ_CHANNELS)
        keep_channels = self.cfg.get('FREQ_KEEP_CHANNELS', None)
        expected_channels = expected_input_channels(
            mode=preprocess_mode,
            keep_channels=keep_channels,
            low_freq_channels=low_freq_channels,
        )
        configured_channels = self.cfg.get('INPUT_CHANNELS', expected_channels)
        if configured_channels != expected_channels:
            raise RuntimeError(
                "INPUT_CHANNELS={} does not match preprocess setting {}, expected {}".format(
                    configured_channels,
                    preprocess_mode,
                    expected_channels,
                )
            )
        self.cfg['INPUT_CHANNELS'] = configured_channels
        os.makedirs(self.cfg['MODEL_ROOT'], exist_ok=True)
        self.make_inputs()
        self.make_model()
        self.maybe_add_channel_gate()
        self.backbone.cuda()
        self.loss = get_loss('DistCrossEntropy').cuda()
        self.opt = self.get_optimizer()
        self.register_hooks()
        self.pfc = self.cfg['HEAD_NAME'] == 'PartialFC'

    def train(self):
        """
        make inputs
            |
        make model
            |
        make loss function
            |
        make optimizer
            |
        make auto mix precision grad scalar
            |
        register hooks
            |
        Distributed Data Parallel mode
            |
        loop_step
        """
        self.prepare()
        self.call_hook("before_run")
        self.backbone = DistributedDataParallel(self.backbone, device_ids=[self.local_rank])
        for epoch in range(self.start_epoch, self.epoch_num):
            self.call_hook("before_train_epoch", epoch)
            self.loop_step(epoch)
            self.call_hook("after_train_epoch", epoch)
        self.call_hook("after_run")


def main():
    task_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.environ.get('TRAIN_CONFIG', os.path.join(task_dir, 'train.yaml'))
    task = TrainTask(config_file)
    task.init_env()
    task.train()


if __name__ == '__main__':
    main()
