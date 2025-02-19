import itertools

import torch
import os
import random

import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets.folder import make_dataset
from torchvision import transforms as t
from torchvision.transforms import v2
import numpy as np

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

def set_seed(seed=None, seed_torch=True):
  """
  Function that controls randomness. NumPy and random modules must be imported.

  Args:
    seed : Integer
      A non-negative integer that defines the random state. Default is `None`.
    seed_torch : Boolean
      If `True` sets the random seed for pytorch tensors, so pytorch module
      must be imported. Default is `True`.

  Returns:
    Nothing.
  """
  if seed is None:
    seed = np.random.choice(2 ** 32)
  random.seed(seed)
  np.random.seed(seed)
  if seed_torch:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
  print(f'Random seed {seed} has been set.')

def get_datasets(root, splits, epoch_size=None, frame_transform=None, video_transform=None, clip_len=16, seed=2024):
    train_split = splits[0]
    val_split = splits[1]
    test_split = splits[2]

    samples = get_samples(root)

    set_seed(seed, seed_torch=True)
    random.shuffle(samples)

    start, end = train_split
    train_samples = samples[start:end]
    start, end = val_split
    val_samples = samples[start:end]
    start, end = test_split
    test_samples = samples[start:end]

    train_dataset = VideosDataset(train_samples, frame_transform=frame_transform, video_transform=video_transform, clip_len=clip_len)
    val_dataset = VideosDataset(val_samples, frame_transform=frame_transform, video_transform=video_transform, clip_len=clip_len)
    test_dataset = VideosDataset(test_samples, frame_transform=frame_transform, video_transform=video_transform, clip_len=clip_len)

    return train_dataset, val_dataset, test_dataset



class VideosDataset(torch.utils.data.IterableDataset):
    def __init__(self, samples, epoch_size=None, frame_transform=None, video_transform=None, clip_len=16):
        super(VideosDataset).__init__()
        self.samples = samples

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

    def __pad_video(self, video_frames):
        """Prepad video frames to match clip length."""
        n = len(video_frames)
        if n == self.clip_len:
            return video_frames

        # Create zero frames
        pad_tensor = torch.zeros_like(video_frames[0])
        pad_frames = [pad_tensor] * (self.clip_len - n)  # List of zero frames

        return pad_frames + video_frames  # Prepadding at the beginning

    def __iter__(self):
        for i in range(self.epoch_size):
            # Get random sample
            path, target = random.choice(self.samples)
            # Get video object
            vid = torchvision.io.VideoReader(path, "video")
            metadata = vid.get_metadata()
            video_frames = []  # video frame buffer

            for frame in itertools.islice(vid, self.clip_len):
                video_frames.append(self.frame_transform(frame['data']))
            video_frames = self.__pad_video(video_frames)
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

dataset = VideosDataset("./dataset", epoch_size=None, frame_transform=frame_transform)
loader = DataLoader(dataset, batch_size=12)
