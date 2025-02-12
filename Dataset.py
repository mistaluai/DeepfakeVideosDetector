import itertools

import torch
import os
import random

import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets.folder import make_dataset
from torchvision import transforms as t
from torchvision.transforms import v2


def get_samples(root, extensions=(".mp4", ".avi")):
    samples = []

    # Define class labels
    class_to_idx = {
        "DFD_original_sequences": 0,  # Real videos
        "DFD_manipulated_sequences": 1  # Deepfake videos
    }

    for class_name, label in class_to_idx.items():
        class_dir = os.path.join(root, class_name)
        if class_name == 'DFD_manipulated_sequences':
            class_dir = os.path.join(class_dir, class_name)
        print(class_dir)
        if not os.path.exists(class_dir):
            continue

        # Get all video files in the directory
        for filename in os.listdir(class_dir):
            if filename.endswith(extensions):
                file_path = os.path.join(class_dir, filename)
                samples.append((file_path, label))

    return samples


class RandomDataset(torch.utils.data.IterableDataset):
    def __init__(self, root, epoch_size=None, frame_transform=None, video_transform=None, clip_len=16, split=(0,-1)):
        super(RandomDataset).__init__()
        start, end = split
        self.samples = get_samples(root)[start:end]

        # Allow for temporal jittering
        if epoch_size is None:
            epoch_size = len(self.samples)
        self.epoch_size = epoch_size

        self.clip_len = clip_len
        if frame_transform is None:
            self.frame_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Resize(255),
                                 v2.CenterCrop(224)])
        else:
            self.frame_transform = frame_transform

        self.video_transform = video_transform

    def __len__(self):
        return self.epoch_size

    def __iter__(self):
        for i in range(self.epoch_size):
            # Get random sample
            path, target = random.choice(self.samples)
            # Get video object
            vid = torchvision.io.VideoReader(path, "video")
            metadata = vid.get_metadata()
            video_frames = []  # video frame buffer

            # Seek and return frames
            max_seek = metadata["video"]['duration'][0] - (self.clip_len / metadata["video"]['fps'][0])
            start = random.uniform(0., max_seek)
            for frame in itertools.islice(vid.seek(start), self.clip_len):
                video_frames.append(self.frame_transform(frame['data']))
                current_pts = frame['pts']
            # Stack it into a tensor
            video = torch.stack(video_frames, 0)
            if self.video_transform:
                video = self.video_transform(video)
            output = {
                'video': video,
                'target': target
            }
            yield output



## Example
transforms = [v2.Resize((224, 224))]
frame_transform = t.Compose(transforms)

dataset = RandomDataset("./dataset", epoch_size=None, frame_transform=frame_transform)
loader = DataLoader(dataset, batch_size=12)
