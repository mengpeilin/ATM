import argparse
import glob
import os
from pathlib import Path

from natsort import natsorted


def reset_symlink_folder(folder):
    """Create an empty split directory without deleting real files accidentally."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for entry in folder.iterdir():
        if not entry.is_symlink():
            raise ValueError(f"refusing to remove non-symlink split entry: {entry}")
        entry.unlink()


def split_pretrain_dataset(files, train_folder, val_folder, train_ratio):
    num_files = len(files)
    num_train = int(num_files * train_ratio)
    train_files = files[:num_train]
    val_files = files[num_train:]
    reset_symlink_folder(train_folder)
    reset_symlink_folder(val_folder)

    for f in train_files:
        Path(train_folder, Path(f).name).symlink_to(os.path.relpath(f, train_folder))

    for f in val_files:
        Path(val_folder, Path(f).name).symlink_to(os.path.relpath(f, val_folder))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='./data/atm_libero/')
    parser.add_argument('--train_ratio', type=float, default=0.95)
    args = parser.parse_args()

    for suite_name in os.listdir(args.folder):
        for task_name in os.listdir(os.path.join(args.folder, suite_name)):
            root_dir = os.path.join(args.folder, suite_name, task_name)
            files = glob.glob(os.path.join(root_dir, '*.hdf5'))
            if len(files) == 0:
                raise ValueError('No .hdf5 files found in {}'.format(args.folder))

            files = natsorted(files)

            train_folder = os.path.join(root_dir, 'train')
            val_folder = os.path.join(root_dir, 'val')

            split_pretrain_dataset(files, train_folder, val_folder, args.train_ratio)
