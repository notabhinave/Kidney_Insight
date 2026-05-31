import os
import numpy as np
from tensorflow.keras.utils import Sequence
from src.utils.config import IMG_SIZE, BATCH_SIZE, TUMOUR_LABEL, NON_TUMOUR_LABEL


DATA_DIR = "data/processed_clahe"


class KidneyDataGenerator(Sequence):
    def __init__(self, file_paths, labels, bbox_dict=None,
                 batch_size=BATCH_SIZE, shuffle=True):
        # file_paths: list/array of paths to .npy images
        # labels: corresponding binary labels
        # bbox_dict: optional mapping path->(xmin,ymin,xmax,ymax) normalized
        self.file_paths = file_paths
        self.labels = labels
        self.bbox_dict = bbox_dict or {}
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.file_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[
            index * self.batch_size:(index + 1) * self.batch_size
        ]

        X, y_class, y_box = [], [], []

        for i in batch_indices:
            image = np.load(self.file_paths[i])

            # Normalize
            image = image.astype("float32") / 255.0

            # Convert to 3 channels
            image = np.stack([image, image, image], axis=-1)

            X.append(image)
            y_class.append(self.labels[i])

            path = self.file_paths[i]
            if path in self.bbox_dict:
                y_box.append(self.bbox_dict[path])

        X = np.array(X)
        y_class = np.array(y_class)

        if y_box:
            y_box = np.array(y_box)
            return X, {"class_output": y_class, "bbox_output": y_box}
        else:
            return X, y_class

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
