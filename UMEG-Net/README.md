# UMEG-Net

Precise event spotting (PES) aims to recognize fine-grained events at exact moments and has become a key component of sports analytics. This task is particularly challenging due to rapid succession, motion blur, and subtle visual differences. Consequently, most existing methods rely on domain-specific, end-to-end training with large labeled datasets and often struggle in few-shot conditions due to their dependence on pixel- or pose-based inputs alone. However, obtaining large labeled datasets is practically hard. We propose a Unified Multi-Entity Graph Network (UMEG-Net) for few-shot PES. UMEG-Net integrates human skeletons and sport-specific object keypoints into a unified graph and features an efficient spatio-temporal extraction module based on advanced GCN and multi-scale temporal shift. To further enhance performance, we employ multimodal distillation to transfer knowledge from keypoint-based graphs to visual representations. Our approach achieves robust performance with limited labeled data and significantly outperforms baseline models in few-shot settings, providing a scalable and effective solution for few-shot PES. Code is publicly available.

## Environment
The code is tested in Linux (Ubuntu 22.04) with the dependency versions in requirements.txt.

## Dataset
Refer to the READMEs in the data directory for pre-processing and setup instructions.

## Training
To train the UMEG-Net model, use `python3 train_umeg-net.py <dataset_name> --frame_dir=<frame_dir> --feat_dir=<pose_dir> --stage=<training_stage> --visual_arch=<visual_arch> --skeleton_arch=<skeleton_arch> --num_samples=<num_samples> -s <save_dir>`.

* `<dataset_name>`: name of the dataset (i.e., f3set-tennis, shuttleset, finegym-BB, fs_comp, soccernet_ball)
* `<frame_dir>`: path to the extracted frames
* `<feat_dir>`: path to extracted keypoints
* `<training_stage>`: training stage (i.e., 1, 2, or 3)
* `<visual_arch>`: visual-based (RGB and optical flow) feature extractor architecture (e.g., videomaev2, rny002, rny002_tsm)
* `<num_samples>`: number of sample clips used for training in few-shot settings (e.g., 25, 100-clip); if -1, use all training samples
* `<save_dir>`: path to save logs, checkpoints, and inference results; please include "stageX" in the saved directory name (e.g., "stage1", "stage2")

Training will produce checkpoints, predictions for the `val` split, and predictions for the `test` split on the best validation epoch.

## Data format
Each dataset has plaintext files that contain the list of event types `events.txt` and elements: `elements.txt`

This is a list of the event names, one per line: `{split}.json`

This file contains entries for each video and its contained events.

```
[
    {
        "fps": 25,
        "height": 720,
        "width": 1280,
        "num_frames": 342,  // number of frames in this clip
        "video": "20210909-W-US_Open-SF-Aryna_Sabalenka-Leylah_Fernandez_170943_171285",  // "video name"_"start frame of the clip"_"end frame of the clip"
        "events": [
            {
                "frame": 100,               // Frame
                "label": EVENT_NAME,        // Event type
            },
            ...
        ],
    },
    ...
]
```

**Frame directory**

We assume pre-extracted frames, that have been resized to 224 pixels high or similar. The organization of the frames is expected to be <frame_dir>/<video_id>/<frame_number>.jpg. For example,

```
video1/
├─ 000000.jpg
├─ 000001.jpg
├─ 000002.jpg
├─ ...
video2/
├─ 000000.jpg
├─ ...
```

Similar format applies to the frames containing objects of interest.

**Flow directory**

We assume pre-extracted optical flows, that have been resized to 224 pixels high or similar. The organization of the frames is expected to be <flow_dir>/<video_id>/<flow_number>.jpg. For example,

```
video1/
├─ 000000.jpg
├─ 000001.jpg
├─ 000002.jpg
├─ ...
video2/
├─ 000000.jpg
├─ ...
```

The feature keypoints for 2D human poses, ball keypoints (if any), and court corners (if any) for all datasets will be uploaded soon.
