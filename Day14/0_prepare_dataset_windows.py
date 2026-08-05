r"""
Day 14 - ONE-TIME dataset preparation (Windows only)
====================================================

Run this ONCE before the other scripts, but only if you are on Windows:

    python 0_prepare_dataset_windows.py

WHY THIS FILE EXISTS
--------------------
On Windows, `tfds.load("cats_vs_dogs")` downloads its 786 MB zip perfectly
well and then crashes while unpacking it:

    KeyError: "There is no item named 'PetImages\\Cat\\0.jpg' in the archive"

This is a bug in TensorFlow Datasets itself, not in our code. Inside
`tensorflow_datasets/image_classification/cats_vs_dogs.py` it does:

    norm_fname = os.path.normpath(fname)      # 'PetImages/Cat/0.jpg'
                                              #   -> 'PetImages\\Cat\\0.jpg' on Windows
    new_zip.writestr(norm_fname, ...)         # zipfile stores it back as '.../...' with '/'
    zipfile.ZipFile(buffer).open(norm_fname)  # looks it up with '\' -> KeyError

On Linux and macOS `os.path.normpath` leaves the forward slashes alone, so
the bug never shows up there. On Windows it renames the file on the way in
and then cannot find it on the way out.

THE FIX
-------
Give that one module a `normpath` that turns every backslash into a forward
slash, which is what the zip actually stores. Nothing else changes, and the
regex TFDS uses to read the labels already accepts both slash styles:

    _NAME_RE = re.compile(r"^PetImages[\\/](Cat|Dog)[\\/]\d+\.jpg$")

We only patch the copy of the module living in memory, for the length of
this one script. No files in site-packages are edited.

After this finishes, the dataset is cached in
`~/tensorflow_datasets/cats_vs_dogs/` and every other script in this folder
can call plain `tfds.load("cats_vs_dogs", ...)` with no patch at all.

HOW LONG IT TAKES
-----------------
About 15 minutes: it re-encodes all 23,262 usable JPEGs and writes them into
TFRecord shards. The 786 MB download is cached, so if you already downloaded
it once it will not download again.
"""

import os as _os
import posixpath
import sys

_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow_datasets as tfds
import tensorflow_datasets.image_classification.cats_vs_dogs as cats_vs_dogs_module


class _ForwardSlashPath:
    """Pretends to be `os.path`, but `normpath` always returns '/' separators."""

    @staticmethod
    def normpath(path):
        return posixpath.normpath(path.replace("\\", "/"))

    def __getattr__(self, name):
        return getattr(_os.path, name)


class _PatchedOS:
    """Pretends to be the `os` module, with only `os.path.normpath` changed."""

    path = _ForwardSlashPath()

    def __getattr__(self, name):
        return getattr(_os, name)


def main():
    if _os.name != "nt":
        print("You are not on Windows - you do not need this script.")
        print('Just run: tfds.load("cats_vs_dogs", with_info=True, as_supervised=True)')

    print("Patching tensorflow_datasets.image_classification.cats_vs_dogs ...")
    cats_vs_dogs_module.os = _PatchedOS()

    print("Building the dataset (this takes ~15 minutes, be patient) ...\n")
    dataset, info = tfds.load(
        "cats_vs_dogs",
        with_info=True,
        as_supervised=True,
    )

    print("\n" + "=" * 66)
    print("SUCCESS - the dataset is ready")
    print("=" * 66)
    print(f"Images   : {info.splits['train'].num_examples:,}")
    print(f"Classes  : {info.features['label'].names}")
    print(f"Cached in: {info.data_dir}")
    print("\nNow run:")
    print("  python 1_transfer_learning_practice.py")
    print("  python 2_cats_vs_dogs_transfer_learning.py")


if __name__ == "__main__":
    sys.exit(main())
