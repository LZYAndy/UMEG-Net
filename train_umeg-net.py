#!/usr/bin/env python3
""" Training for UMEG-Net """
import os
import argparse
from contextlib import nullcontext
import random
import numpy as np
import torch
import math

torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import (
    ChainedScheduler, LinearLR, CosineAnnealingLR)
from torch.utils.data import DataLoader
import torchvision
from itertools import groupby
import timm
from tqdm import tqdm

from model.common import step, BaseRGBModel
from model.shift import make_temporal_shift
from model.vitmae import VisionTransformer
from model.umegnet import UMEG
from model.modules import *
from dataset.input_process import ActionSeqDataset, ActionSeqVideoDataset
from util.eval import edit_score, non_maximum_suppression, non_maximum_suppression_np
from util.io import load_json, store_json, clear_files
from util.dataset import DATASETS, load_classes
from util.score import compute_mAPs
from thop import profile
import warnings

warnings.filterwarnings("ignore")

BASE_NUM_WORKERS = 4
BASE_NUM_VAL_EPOCHS = 20
INFERENCE_BATCH_SIZE = 4
HIDDEN_DIM = 368


def get_args():
    """
    Parses command-line arguments for model training and evaluation.
    Returns an argparse.Namespace containing all arguments.
    """
    parser = argparse.ArgumentParser()

    # Required positional argument for dataset name (must be in DATASETS)
    parser.add_argument('dataset', type=str, choices=DATASETS)

    # Optional: Path to frame and keypoint feature directories
    parser.add_argument('--frame_dir', type=str, default=None, help='Path to extracted frames')
    parser.add_argument('--feat_dir', type=str, default=None, help='Path to extracted keypoints')

    # Stage of training pipeline (1: pretrain, 2: distill, 3: finetune)
    parser.add_argument('--stage', type=int, required=True, help='Training stage (1, 2, or 3)')

    # Visual backbone architecture to use for feature extraction
    parser.add_argument(
        '--visual_arch', type=str, choices=[
            'rn50', 'rn50_tsm', 'rn50_gsm',
            'rny002', 'rny002_tsm', 'rny002_gsm',
            'rny008', 'rny008_tsm', 'rny008_gsm',
            'videomaev2',
        ], default='videomaev2', help='Architecture for feature extraction'
    )

    # Temporal model architecture (e.g., GRU or deeper GRU)
    parser.add_argument(
        '-t', '--temporal_arch', type=str, default='gru',
        choices=['gru', 'deeper_gru']
    )

    # Few-shot setting: number of samples per class (if -1, use all)
    parser.add_argument('--num_samples', type=int, default=-1,
                        help='few-shot k-clip samples, -1 means all samples')
    parser.add_argument('--clip_len', type=int, default=96)
    parser.add_argument('--crop_dim', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--window', type=int, default=5, help='NMS window size')
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('-ag', '--acc_grad_iter', type=int, default=1, help='Gradient accumulation')

    # Training schedule
    parser.add_argument('--warm_up_epochs', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-s', '--save_dir', type=str, required=True, help='Directory to save checkpoints')

    # Training resume and validation
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--start_val_epoch', type=int, default=30)
    parser.add_argument('--criterion', choices=['edit', 'loss'], default='edit')
    parser.add_argument('--dilate_len', type=int, default=0, help='Label dilation')

    # DataLoader and device settings
    parser.add_argument('-j', '--num_workers', type=int, help='Number of dataloader workers')
    parser.add_argument('-mgpu', '--gpu_parallel', action='store_true')

    return parser.parse_args()


class UMEGNet(BaseRGBModel):
    """
    Unified Multi-Entity Graph Network (UMEG-Net) model for precise event spotting in sports videos.
    Wraps feature extraction, temporal modeling, and event prediction for both RGB and skeleton modalities.
    """

    class Impl(nn.Module):
        """
        The actual PyTorch model implementation. Contains network modules for feature extraction,
        temporal modeling, and event prediction.
        """

        def __init__(self, num_classes, visual_arch, temporal_arch, clip_len, stage=1, step=1, window=5, device='cuda',
                     dataset_name=None):
            super().__init__()
            self._device = device
            self._num_classes = num_classes
            self._window = window
            self._stage = stage
            self._visual_arch = visual_arch

            # Choose RGB visual backbone based on visual_arch argument
            if 'rn50' in visual_arch:
                # Use torchvision's ResNet-50
                resnet_name = visual_arch.split('_')[0].replace('rn', 'resnet')
                rgb_feat = getattr(torchvision.models, resnet_name)(pretrained=True)
                rgb_feat_dim = rgb_feat.fc.in_features
                rgb_feat.fc = nn.Identity()  # Remove classification head

            elif visual_arch.startswith(('rny002', 'rny008')):
                # Use TIMM's RegNetY models
                rgb_feat = timm.create_model({
                                                 'rny002': 'regnety_002',
                                                 'rny008': 'regnety_008',
                                             }[visual_arch.rsplit('_', 1)[0]], pretrained=True)
                rgb_feat_dim = rgb_feat.head.fc.in_features
                rgb_feat.head.fc = nn.Identity()  # Remove classification head

            elif 'videomaev2' in visual_arch:
                # Use a pretrained VideoMAEv2 transformer
                model_path = './model/vit_b_k710_dl_from_giant.pth'
                rgb_feat = VisionTransformer(embed_dim=768, depth=12, num_heads=12, all_frames=2)
                checkpoint_3d = torch.load(model_path, map_location='cpu')
                rgb_feat.load_state_dict(checkpoint_3d['module'], strict=False)
                rgb_feat_dim = 768
                rgb_feat.head = nn.Identity()

            else:
                raise NotImplementedError(visual_arch)

            # Set number of persons/entities based on dataset type
            num_person = 2 if 'f3set' in dataset_name else 22 if 'soccer' in dataset_name else 1

            # Instantiate the UMEG skeleton-based feature extractor
            umeg_feat = UMEG(in_channels=2, num_person=num_person, steps=[1, 2, 4], gcn_with_res=True)
            umeg_feat_dim = 256

            # Optionally add Temporal Shift/GSM to visual backbone
            self._require_clip_len = clip_len
            if '_tsm' in visual_arch:
                make_temporal_shift(rgb_feat, clip_len, is_gsm=False, step=step)
            elif '_gsm' in visual_arch:
                make_temporal_shift(rgb_feat, clip_len, is_gsm=True)

            self._rgb_feat = rgb_feat  # RGB feature extractor module
            self._umeg_feat = umeg_feat  # Skeleton feature extractor

            # Choose temporal model depth
            d_model = HIDDEN_DIM
            if temporal_arch == 'gru':  # Single layer GRU
                self._rgb_temporal = GRU(rgb_feat_dim, d_model, num_layers=1)
                self._umeg_temporal = GRU(umeg_feat_dim, d_model, num_layers=1)
            elif temporal_arch == 'deeper_gru':  # Multi-layer GRU
                self._rgb_temporal = GRU(rgb_feat_dim, d_model, num_layers=3)
                self._umeg_temporal = GRU(umeg_feat_dim, d_model, num_layers=3)
            else:
                raise NotImplementedError(temporal_arch)

            # Freeze feature extractors for later stages if needed
            if stage > 1:
                for param in self._umeg_feat.parameters():
                    param.requires_grad = False
                for param in self._umeg_temporal.parameters():
                    param.requires_grad = False
                if stage > 2:
                    for param in self._rgb_feat.parameters():
                        param.requires_grad = False

            # Prediction heads
            self._coarse_pred = nn.Linear(d_model, 2)  # Binary coarse event detection (foreground/background)
            self._fine_pred = nn.Linear(d_model, num_classes)  # Multi-label fine-grained event classifier

        def forward(self, frame, skeleton, umeg_feature):
            """
            Forward pass for the UMEGNet model.
            frame: RGB video frames
            skeleton: Keypoint sequences (skeletons)
            umeg_feature: Precomputed features for UMEG
            Returns:
                coarse_pred, fine_pred, rgb_feat, umeg_feat
            """
            rgb_feat, umeg_feat = None, None
            enc_feat = None

            # Extract visual (RGB) features if provided
            if frame is not None:
                batch_size, clip_len, channels, height, width = frame.shape
                # Use normal backbone or VideoMAE for feature extraction
                if 'mae' not in self._visual_arch:
                    rgb_feat = self._rgb_feat(frame.view(-1, channels, height, width)).reshape(batch_size, clip_len,
                                                                                               -1)
                else:
                    # For MAE: special reshaping due to frame grouping
                    rgb_input = frame.reshape(batch_size, clip_len // 2, 2, channels, height, width)
                    rgb_input = rgb_input.reshape(batch_size * clip_len // 2, 2, channels, height, width)
                    rgb_feat = self._rgb_feat(rgb_input.transpose(1, 2))
                    rgb_feat = rgb_feat.view(batch_size, -1, 768).permute(0, 2, 1)
                    rgb_feat = F.interpolate(rgb_feat, scale_factor=2, mode='linear', align_corners=False).permute(0,
                                                                                                                   2,
                                                                                                                   1)

                # Temporal modeling on RGB features
                rgb_feat = self._rgb_temporal(rgb_feat)
                enc_feat = rgb_feat

            # Extract skeleton-based features if provided
            if skeleton is not None and umeg_feature is not None:
                umeg_feat = self._umeg_feat(skeleton.transpose(1, 2), umeg_feature)
                umeg_feat = self._umeg_temporal(umeg_feat)
                # If only skeleton is provided, use its encoded feature
                if frame is None:
                    enc_feat = umeg_feat

            # Event localization and classification
            coarse_pred = self._coarse_pred(enc_feat)  # Binary
            fine_pred = self._fine_pred(enc_feat)  # Multi-label

            return coarse_pred, fine_pred, rgb_feat, umeg_feat

    def __init__(self, num_classes, visual_arch, temporal_arch, clip_len, step=1, window=5, stage=1, device='cuda',
                 multi_gpu=False, dataset_name=None):
        """
        Initializes the UMEGNet high-level interface (handles DataParallel, device transfer, etc.).
        """
        self._device = device
        self._multi_gpu = multi_gpu
        self._window = window
        self._stage = stage
        # Construct the actual network (Impl)
        self._model = UMEGNet.Impl(num_classes, visual_arch, temporal_arch, clip_len, step=step,
                                   window=window, stage=stage, dataset_name=dataset_name)
        # Optionally wrap for multi-GPU
        if multi_gpu:
            self._model = nn.DataParallel(self._model)
        self._model.to(device)
        self._num_classes = num_classes

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None, acc_grad_iter=1, fg_weight=5):
        """
        One full training/validation epoch.
        loader: DataLoader
        optimizer: torch.optim.Optimizer (if None, run in eval mode)
        scaler: torch.cuda.amp.GradScaler for mixed-precision training
        lr_scheduler: optional LR scheduler
        acc_grad_iter: gradient accumulation steps
        fg_weight: weight for foreground class in coarse prediction
        Returns the average loss.
        """
        if optimizer is None:
            self._model.eval()
        else:
            optimizer.zero_grad()
            self._model.train()

        # Class weight for foreground/background in coarse prediction
        ce_kwargs = {}
        if fg_weight != 1:
            ce_kwargs['weight'] = torch.FloatTensor([1, fg_weight]).to(self._device)

        epoch_loss = 0.
        # Enable torch.no_grad during evaluation
        with (torch.no_grad() if optimizer is None else nullcontext()):
            for batch_idx, batch in enumerate(tqdm(loader)):
                # Move batch data to device
                frame = loader.dataset.load_frame_gpu(batch, self._device)
                skeleton = loader.dataset.load_skeleton_gpu(batch, self._device)
                umeg_feature = loader.dataset.load_feature_gpu(batch, self._device)

                # Labels and masks
                coarse_label = batch['coarse_label'].to(self._device)
                fine_label = batch['fine_label'].float().to(self._device)
                coarse_mask = batch['coarse_mask'].to(self._device)

                with torch.cuda.amp.autocast():
                    loss = 0.

                    # Stage 1: pretraining UMEG
                    if self._stage == 1:
                        coarse_pred, fine_pred, _, _ = self._model(None, skeleton, umeg_feature)

                    # Stage 2: multimodal distillation
                    elif self._stage == 2:
                        _, _, rgb_feat, umeg_feat = self._model(frame, skeleton, umeg_feature)
                        # L2 distance between RGB and skeleton features
                        l2_loss = F.mse_loss(rgb_feat, umeg_feat)
                        loss += l2_loss

                    # Stage 3: few-shot finetuning on RGB only
                    elif self._stage == 3:
                        coarse_pred, fine_pred, _, _ = self._model(frame, None, None)

                    # For stages except distillation, compute event localization/classification losses
                    if self._stage != 2:
                        # Coarse-grained (foreground/background) event loss
                        coarse_loss = F.cross_entropy(coarse_pred.reshape(-1, 2), coarse_label.flatten(), **ce_kwargs)
                        if not math.isnan(coarse_loss.item()):
                            loss += coarse_loss

                        # Fine-grained multi-label event classification loss (masked)
                        fine_bce_loss = F.binary_cross_entropy_with_logits(fine_pred, fine_label.float(),
                                                                           reduction='none')
                        fine_bce_loss = fine_bce_loss * coarse_mask.unsqueeze(2).expand_as(fine_pred)
                        fine_mask = coarse_label.unsqueeze(2).expand_as(fine_pred)
                        masked_fine_loss = fine_bce_loss * fine_mask
                        fine_loss = masked_fine_loss.sum() / fine_mask.sum()
                        if not math.isnan(fine_loss.item()):
                            loss += fine_loss

                # Backward pass and optimization if training
                if optimizer is not None and loss != 0.:
                    step(optimizer, scaler, loss / acc_grad_iter, lr_scheduler=lr_scheduler,
                         backward_only=(batch_idx + 1) % acc_grad_iter != 0)

                if loss != 0.:
                    epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)

    def predict(self, frame, skeleton=None, umeg_feature=None, use_amp=True):
        """
        Inference on a single clip/sample.
        frame: RGB frames [B, T, C, H, W] or [T, C, H, W]
        skeleton: optional keypoints
        umeg_feature: optional UMEG features
        use_amp: whether to use mixed precision
        Returns: (coarse_pred_cls, coarse_pred, fine_pred) as numpy arrays.
        """
        # Prepare inputs
        if not isinstance(frame, torch.Tensor):
            frame = torch.FloatTensor(frame)
        if len(frame.shape) == 4:  # (L, C, H, W)
            frame = frame.unsqueeze(0)
        frame = frame.to(self._device)

        if skeleton is not None:
            if not isinstance(skeleton, torch.Tensor):
                skeleton = torch.FloatTensor(skeleton)
            if len(skeleton.shape) == 4:  # (L, C, H, W)
                skeleton = skeleton.unsqueeze(0)
            skeleton = skeleton.to(self._device)

        if umeg_feature is not None:
            if not isinstance(umeg_feature, torch.Tensor):
                umeg_feature = torch.FloatTensor(umeg_feature)
            if len(umeg_feature.shape) == 4:  # (L, C, H, W)
                umeg_feature = umeg_feature.unsqueeze(0)
            umeg_feature = umeg_feature.to(self._device)

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast() if use_amp else nullcontext():
                # Depending on stage, select which modality to use
                if self._stage == 1:
                    coarse_pred, fine_pred, _, _ = self._model(None, skeleton, umeg_feature)
                else:
                    coarse_pred, fine_pred, _, _ = self._model(frame, None, None)

            # Softmax and NMS for coarse predictions
            coarse_pred = torch.softmax(coarse_pred, axis=2)
            coarse_pred = non_maximum_suppression(coarse_pred, self._window)
            coarse_pred_cls = torch.argmax(coarse_pred, axis=2)
            fine_pred = torch.sigmoid(fine_pred)

            return coarse_pred_cls.cpu().numpy(), coarse_pred.cpu().numpy(), fine_pred.cpu().numpy()


def evaluate(model, dataset, classes, delta=1, window=5, dataset_name='f3set-tennis', device='cuda'):
    """
    Evaluate the event spotting model on the provided dataset.
    Args:
        model: Model to evaluate.
        dataset: Dataset object.
        classes: Dictionary mapping class names to indices.
        delta: Tolerance window for event match in F1 calculation.
        window: NMS window size (not used here, kept for compatibility).
        dataset_name: Name of the dataset for dataset-specific logic.
        device: Computation device.
    Returns:
        Mean edit score across all videos.
    """
    # Initialize a prediction dictionary for each video: stores
    # [coarse scores, fine scores, and count of predictions/support]
    pred_dict = {}
    for video, video_len, _ in dataset.videos:
        pred_dict[video] = (
            np.zeros((video_len, 2), np.float32),  # Coarse (binary) scores per frame
            np.zeros((video_len, len(classes)), np.float32),  # Fine (multi-class) scores per frame
            np.zeros(video_len, np.int32))  # Support count per frame

    # Map class index to class name for reporting
    classes_inv = {v: k for k, v in classes.items()}
    classes_inv[0] = 'NA'  # Add a placeholder for background

    # Set batch size for inference (single for augmentation)
    batch_size = 1 if dataset.augment else INFERENCE_BATCH_SIZE

    # Iterate over the dataset in batches for prediction
    for clip in tqdm(DataLoader(
            dataset, num_workers=BASE_NUM_WORKERS * 2, pin_memory=True,
            batch_size=batch_size
    )):
        if batch_size > 1:
            # Predict for each clip in the batch
            _, batch_coarse_scores, batch_fine_scores = model.predict(
                clip['frame'], clip['skeleton'], clip['feature']
            )
            # For each sample in the batch, accumulate its prediction into pred_dict
            for i in range(clip['frame'].shape[0]):
                video = clip['video'][i]
                coarse_scores, fine_scores, support = pred_dict[video]
                coarse_pred_scores = batch_coarse_scores[i]
                fine_pred_scores = batch_fine_scores[i]

                # Determine clip's placement in the full video
                start = clip['start'][i].item()
                # Handle clips that start before the beginning
                if start < 0:
                    coarse_pred_scores = coarse_pred_scores[-start:, :]
                    fine_pred_scores = fine_pred_scores[-start:, :]
                    start = 0
                end = start + coarse_pred_scores.shape[0]
                # Truncate predictions that exceed video length
                if end >= coarse_scores.shape[0]:
                    end = coarse_scores.shape[0]
                    coarse_pred_scores = coarse_pred_scores[:end - start, :]
                    fine_pred_scores = fine_pred_scores[:end - start, :]
                # Accumulate predictions and increment support
                coarse_scores[start:end, :] += coarse_pred_scores
                fine_scores[start:end, :] += fine_pred_scores
                support[start:end] += 1

    # Start evaluation: open file to record error sequences for analysis
    f = open('error_sequences.txt', 'w')
    edit_scores = []
    f1_event = dict()  # Store counts for F1 calculation

    # For each video, post-process predictions and calculate metrics
    for video, (coarse_scores, fine_scores, support) in sorted(pred_dict.items()):
        # Load ground truth labels
        coarse_label, fine_label = dataset.get_labels(video)
        # Average accumulated scores over number of predictions/support
        coarse_scores /= support[:, None]
        fine_scores /= support[:, None]

        # Argmax over coarse scores gives predicted binary label (0/1) per frame
        coarse_pred = np.argmax(coarse_scores, axis=1)

        # Prepare empty array for fine predictions
        fine_pred = np.zeros_like(fine_scores, int)

        # Dataset-specific logic for decoding fine-grained predictions
        if 'f3set-tennis' in dataset_name:
            for i in range(len(fine_scores)):
                # For each event group (serve, rally, etc.), pick highest score as event prediction
                for start, end in [[0, 2], [2, 5], [5, 8], [16, 24], [25, 29]]:
                    max_idx = np.argmax(fine_scores[i, start:end])
                    fine_pred[i, start + max_idx] = 1
                # Special case for approach (single class)
                if fine_scores[i, 24] > 0.5:
                    fine_pred[i, 24] = 1
                # If frame is not serve, predict other event groups
                if fine_pred[i, 5] != 1:
                    for start, end in [[8, 10], [10, 16]]:
                        max_idx = np.argmax(fine_scores[i, start:end])
                        fine_pred[i, start + max_idx] = 1

        elif 'shuttleset' in dataset_name:
            for i in range(len(fine_scores)):
                for start, end in [[0, 2], [2, 20]]:
                    max_idx = np.argmax(fine_scores[i, start:end])
                    fine_pred[i, start + max_idx] = 1

        elif 'soccernet_ball' in dataset_name:
            delta = 12
            for i in range(len(fine_scores)):
                for start, end in [[0, 12]]:
                    max_idx = np.argmax(fine_scores[i, start:end])
                    fine_pred[i, start + max_idx] = 1

        elif 'finegym-BB' in dataset_name:
            delta = 3
            for i in range(len(fine_scores)):
                for start, end in [[0, 2], [2, 7]]:
                    max_idx = np.argmax(fine_scores[i, start:end])
                    fine_pred[i, start + max_idx] = 1

        elif 'fs_comp' in dataset_name:
            for i in range(len(fine_scores)):
                for start, end in [[0, 2], [2, 12]]:
                    max_idx = np.argmax(fine_scores[i, start:end])
                    fine_pred[i, start + max_idx] = 1

        # Only consider fine predictions for frames predicted as foreground
        fine_pred = coarse_pred[:, np.newaxis] * fine_pred

        # Prepare lists to print event prediction/ground-truth sequences for error analysis
        print_preds, print_gts = [], []
        for i in range(len(fine_pred)):
            if coarse_label[i] == 1:
                print_gt = []
                for j in range(len(fine_pred[0])):
                    if fine_label[i, j] == 1:
                        print_gt.append(classes_inv[j + 1])
                print_gts.append('_'.join(print_gt))
            if coarse_pred[i] == 1:
                print_pred = []
                for j in range(len(fine_pred[0])):
                    if fine_pred[i, j] == 1:
                        print_pred.append(classes_inv[j + 1])
                print_preds.append('_'.join(print_pred))

        # Convert fine-grained multi-hot vectors to integers for sequence matching
        labels = [int(''.join(str(x) for x in row), 2) for row in fine_label]
        preds = [int(''.join(str(x) for x in row), 2) for row in fine_pred]
        preds = coarse_pred * preds  # Only keep predictions where foreground is predicted

        # Event-level F1 calculation (matches in a window)
        for i in range(len(preds)):
            # True positive: predicted event matches any ground truth within delta
            if preds[i] > 0 and preds[i] in labels[max(0, i - delta):min(len(preds), i + delta + 1)]:
                if preds[i] not in f1_event:
                    f1_event[preds[i]] = [1, 0, 0]  # [tp, fp, fn]
                else:
                    f1_event[preds[i]][0] += 1
            # False positive: prediction but no matching ground truth nearby
            if preds[i] > 0 and sum(labels[max(0, i - delta):min(len(preds), i + delta + 1)]) == 0:
                if preds[i] not in f1_event:
                    f1_event[preds[i]] = [0, 1, 0]
                else:
                    f1_event[preds[i]][1] += 1
            # False negative: missed ground truth event (no prediction nearby)
            if labels[i] > 0 and labels[i] not in preds[max(0, i - delta):min(len(preds), i + delta + 1)]:
                if labels[i] not in f1_event:
                    f1_event[labels[i]] = [0, 0, 1]
                else:
                    f1_event[labels[i]][2] += 1

        # Collapse repeated events for edit distance evaluation
        gt = [k for k, g in groupby(labels) if k != 0]
        pred = [k for k, g in groupby(preds) if k != 0]

        # Record sequences where prediction and ground truth don't match (for analysis)
        if len(pred) == len(gt):
            for j in range(len(pred)):
                if pred[j] != gt[j]:
                    f.write(video + '\n')
                    f.write('->'.join(print_preds) + '\n\n')
                    f.write('->'.join(print_gts) + '\n\n')
                    f.write('------------------------\n')
                    break
        else:
            f.write(video + '\n')
            f.write('->'.join(print_preds) + '\n\n')
            f.write('->'.join(print_gts) + '\n\n')
            f.write('------------------------\n')

        # Compute edit score (Levenshtein distance, segment-level) for the video
        edit_scores.append(edit_score(pred, gt))

    f.close()

    # Compute mean event F1
    f1, count = 0, 0
    for value in f1_event.values():
        if sum(value) == 0:
            continue
        precision = value[0] / (value[0] + value[1] + 1e-10)
        recall = value[0] / (value[0] + value[2] + 1e-10)
        f1 += 2 * precision * recall / (precision + recall + 1e-10)
        count += 1
    f1 /= count
    print('Mean F1 (event):', np.mean(f1))
    print()
    print('Edit score:', sum(edit_scores) / len(edit_scores))
    return sum(edit_scores) / len(edit_scores)


def get_last_epoch(save_dir):
    max_epoch = -1
    for file_name in os.listdir(save_dir):
        if not file_name.startswith('optim_'):
            continue
        epoch = int(os.path.splitext(file_name)[0].split('optim_')[1])
        if epoch > max_epoch:
            max_epoch = epoch
    return max_epoch


def get_best_epoch_and_history(save_dir, criterion):
    data = load_json(os.path.join(save_dir, 'loss.json'))
    if criterion == 'edit':
        key = 'val_edit'
        best = max(data, key=lambda x: x[key])
    else:
        key = 'val'
        best = min(data, key=lambda x: x[key])
    return data, best['epoch'], best[key]


def get_datasets(args):
    classes = load_classes(os.path.join('data', args.dataset, 'elements.txt'))

    if 'shuttleset' in args.dataset:
        epoch_num_frames = 200000 if (args.stage == 2 or args.num_samples == -1) else 100000
    else:
        epoch_num_frames = 500000 if (args.stage == 2 or args.num_samples == -1) else 100000

    dataset_len = epoch_num_frames // (args.clip_len * args.stride)
    dataset_kwargs = {
        'crop_dim': args.crop_dim, 'stride': args.stride
    }

    print('Dataset size:', dataset_len)
    num_train_samples, num_val_samples = -1, -1
    if args.num_samples > 0:
        num_train_samples = int(args.num_samples * 0.8)
        num_val_samples = args.num_samples - num_train_samples
    train_data = ActionSeqDataset(
        classes, os.path.join('data', args.dataset, 'train.json'),
        args.frame_dir, args.clip_len, dataset_len, is_eval=False, dilate_len=args.dilate_len, stage=args.stage,
        num_samples=num_train_samples, feat_dir=args.feat_dir, **dataset_kwargs)
    train_data.print_info()
    val_data = ActionSeqDataset(
        classes, os.path.join('data', args.dataset, 'val.json'),
        args.frame_dir, args.clip_len, dataset_len // 4, dilate_len=args.dilate_len, stage=args.stage,
        num_samples=num_val_samples, feat_dir=args.feat_dir, **dataset_kwargs)
    val_data.print_info()

    val_data_frames = None
    if args.criterion == 'edit':
        # Only perform edit score evaluation during training if criterion is edit
        val_data_frames = ActionSeqVideoDataset(
            classes, os.path.join('data', args.dataset, 'val.json'),
            args.frame_dir, args.clip_len, overlap_len=0, num_samples=num_val_samples,
            feat_dir=args.feat_dir, **dataset_kwargs)

    return classes, train_data, val_data, None, val_data_frames


def load_from_save(
        args, model, optimizer, scaler, lr_scheduler
):
    assert args.save_dir is not None
    epoch = get_last_epoch(args.save_dir)

    print('Loading from epoch {}'.format(epoch))
    model.load(torch.load(os.path.join(
        args.save_dir, 'checkpoint_{:03d}.pt'.format(epoch))))

    if args.resume:
        opt_data = torch.load(os.path.join(
            args.save_dir, 'optim_{:03d}.pt'.format(epoch)))
        optimizer.load_state_dict(opt_data['optimizer_state_dict'])
        scaler.load_state_dict(opt_data['scaler_state_dict'])
        lr_scheduler.load_state_dict(opt_data['lr_state_dict'])

    losses, best_epoch, best_criterion = get_best_epoch_and_history(
        args.save_dir, args.criterion)
    return epoch, losses, best_epoch, best_criterion


def store_config(file_path, args, num_epochs, classes):
    config = {
        'dataset': args.dataset,
        'num_classes': len(classes),
        'visual_arch': args.visual_arch,
        'temporal_arch': args.temporal_arch,
        'num_samples': args.num_samples,
        'clip_len': args.clip_len,
        'batch_size': args.batch_size,
        'crop_dim': args.crop_dim,
        'window': args.window,
        'stage': args.stage,
        'stride': args.stride,
        'num_epochs': num_epochs,
        'warm_up_epochs': args.warm_up_epochs,
        'learning_rate': args.learning_rate,
        'start_val_epoch': args.start_val_epoch,
        'gpu_parallel': args.gpu_parallel,
        'dilate_len': args.dilate_len
    }
    store_json(file_path, config, pretty=True)


def get_num_train_workers(args):
    n = BASE_NUM_WORKERS * 2
    return min(os.cpu_count(), n)


def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
        args.warm_up_epochs, cosine_epochs))
    return args.num_epochs, ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer,
                          num_steps_per_epoch * cosine_epochs)])


def main(args):
    if args.num_workers is not None:
        global BASE_NUM_WORKERS
        BASE_NUM_WORKERS = args.num_workers

    assert args.batch_size % args.acc_grad_iter == 0
    if args.start_val_epoch is None:
        args.start_val_epoch = args.num_epochs - BASE_NUM_VAL_EPOCHS
    if args.crop_dim <= 0:
        args.crop_dim = None

    classes, train_data, val_data, train_data_frames, val_data_frames = get_datasets(args)

    def worker_init_fn(id):
        random.seed(id + epoch * 100)

    loader_batch_size = args.batch_size // args.acc_grad_iter
    train_loader = DataLoader(
        train_data, shuffle=False, batch_size=loader_batch_size,
        pin_memory=True, num_workers=get_num_train_workers(args),
        prefetch_factor=1, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=loader_batch_size,
        pin_memory=True, num_workers=BASE_NUM_WORKERS,
        worker_init_fn=worker_init_fn)

    model = UMEGNet(len(classes), args.visual_arch, args.temporal_arch, clip_len=args.clip_len, step=args.stride,
                    window=args.window, stage=args.stage, multi_gpu=args.gpu_parallel, dataset_name=args.dataset)
    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})

    # obtain best epoch from previous stage
    if args.stage == 2:
        stage1_save_dir = args.save_dir.replace('stage2', 'stage1')
        losses, best_epoch, best_criterion = get_best_epoch_and_history(stage1_save_dir, args.criterion)
        print('Loading from STAGE 1 epoch {}'.format(best_epoch))
        model_stage1 = torch.load(os.path.join(stage1_save_dir, 'checkpoint_{:03d}.pt'.format(best_epoch)))
        model.load(model_stage1)
    elif args.stage == 3:
        stage2_save_dir = args.save_dir.replace('stage3', 'stage2')
        losses, best_epoch, best_criterion = get_best_epoch_and_history(stage2_save_dir, args.criterion)
        print('Loading from STAGE 2 epoch {}'.format(best_epoch))
        model.load(torch.load(os.path.join(stage2_save_dir, 'checkpoint_{:03d}.pt'.format(best_epoch))))

    # Warmup schedule
    num_steps_per_epoch = len(train_loader) // args.acc_grad_iter
    num_epochs, lr_scheduler = get_lr_scheduler(
        args, optimizer, num_steps_per_epoch)

    losses = []
    best_epoch = None
    best_criterion = 0 if args.criterion == 'edit' else float('inf')
    best_loss, stop_criterion = float('inf'), 0

    epoch = 0
    if args.resume:
        epoch, losses, best_epoch, best_criterion = load_from_save(args, model, optimizer, scaler, lr_scheduler)
        epoch += 1

    # Write it to console
    store_config('/dev/stdout', args, num_epochs, classes)

    for epoch in range(epoch, num_epochs):
        train_loss = model.epoch(train_loader, optimizer, scaler, lr_scheduler=lr_scheduler,
                                 acc_grad_iter=args.acc_grad_iter)
        val_loss = model.epoch(val_loader, acc_grad_iter=args.acc_grad_iter)
        print('[Epoch {}] Train loss: {:0.5f} Val loss: {:0.5f}'.format(
            epoch, train_loss, val_loss))

        val_edit = 0
        if args.criterion == 'loss':
            if val_loss < best_criterion:
                best_criterion = val_loss
                best_epoch = epoch
                print('New best epoch!')
        elif args.criterion == 'edit':
            if epoch >= args.start_val_epoch:
                val_edit = evaluate(model, val_data_frames, classes, window=args.window, dataset_name=args.dataset)
                if args.criterion == 'edit' and val_edit > best_criterion:
                    best_criterion = val_edit
                    best_epoch = epoch
                    print('New best epoch!')
        else:
            print('Unknown criterion:', args.criterion)

        losses.append({
            'epoch': epoch, 'train': train_loss, 'val': val_loss, 'val_edit': val_edit})
        if args.save_dir is not None:
            os.makedirs(args.save_dir, exist_ok=True)
            store_json(os.path.join(args.save_dir, 'loss.json'), losses,
                       pretty=True)
            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir,
                             'checkpoint_{:03d}.pt'.format(epoch)))
            clear_files(args.save_dir, r'optim_\d+\.pt')
            torch.save(
                {'optimizer_state_dict': optimizer.state_dict(),
                 'scaler_state_dict': scaler.state_dict(),
                 'lr_state_dict': lr_scheduler.state_dict()},
                os.path.join(args.save_dir,
                             'optim_{:03d}.pt'.format(epoch)))
            store_config(os.path.join(args.save_dir, 'config.json'),
                         args, num_epochs, classes)

    print('Best epoch: {}\n'.format(best_epoch))

    if args.save_dir is not None:
        model.load(torch.load(os.path.join(
            args.save_dir, 'checkpoint_{:03d}.pt'.format(best_epoch))))

        # Evaluate on hold out splits
        eval_splits = ['test']
        for split in eval_splits:
            split_path = os.path.join(
                'data', args.dataset, '{}.json'.format(split))
            if os.path.exists(split_path):
                split_data = ActionSeqVideoDataset(classes, split_path, args.frame_dir, args.clip_len,
                                                   flow_dir=args.flow_dir, feat_dir=args.feat_dir,
                                                   overlap_len=args.clip_len // 2, crop_dim=args.crop_dim,
                                                   stride=args.stride, is_test=True)
                split_data.print_info()
                evaluate(model, split_data, classes, window=args.window, dataset_name=args.dataset)


if __name__ == '__main__':
    main(get_args())
