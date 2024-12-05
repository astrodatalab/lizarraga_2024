import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Optional, Tuple, Union

class HDF5ImageGenerator(Dataset):
    """Custom HDF5 ImageDataGenerator for PyTorch.

    Generates individual samples of tensor images from HDF5 files with (optional) real-time
    data augmentation.

    Arguments
    ---------
    src : str
        Path of the HDF5 source file.
    X_key : str
        Key of the HDF5 file image tensors dataset.
        Default is "images".
    y_key : str
        Key of the HDF5 file labels dataset.
        Default is "labels".
    classes_key : str
        Key of the HDF5 file dataset containing
        the raw classes.
        Default is None.
    shuffle : bool
        Shuffle images at the end of each epoch.
        Default is True.
    scaler : "std", "norm" or False
        "std" mode means standardization to range [-1, 1]
        with 0 mean and unit variance.
        "norm" mode means normalization to range [0, 1].
        Default is "std".
    y_scaler : "minmax", "std", or False
        Apply scaling to labels. Currently supports "minmax" or "std".
        Default is False.
    y_range : tuple
        Needed when setting y_scaler to "minmax".
        Standardization to the provided range.
        Default is (0, 1).
    num_classes : None or int
        Specifies the total number of classes
        for labels encoding.
        Default is None.
    labels_encoding : "hot", "smooth" or False
        "hot" mode means classic one-hot encoding.
        "smooth" mode means smooth hot encoding.
        Default is "hot".
    smooth_factor : int or float
        Smooth factor used by smooth
        labels encoding.
        Default is 0.1.
    augmenter : albumentations.Compose or False
        An albumentations transformations pipeline
        to apply to each sample.
        Default is False.
    mode : str
        "train", "validation", or "test".
        Default is "train".

    Notes
    -----
    Turn off scaler (scaler=False) if using the
    ToFloat(max_value=255) transformation from
    albumentations.
    """

    def __init__(
        self,
        src: str,
        X_key: str = "images",
        y_key: str = "labels",
        classes_key: Optional[str] = None,
        shuffle: bool = True,
        scaler: Union[str, bool] = "std",
        y_scaler: Union[str, bool] = False,
        y_range: Tuple[float, float] = (0, 1),
        num_classes: Optional[int] = None,
        labels_encoding: Union[str, bool] = "hot",
        smooth_factor: float = 0.1,
        augmenter: Union[bool, object] = False,
        mode: str = "train",
    ):
        # Define available modes
        available_modes = ['train', 'validation', 'test']
        if mode not in available_modes:
            raise ValueError(f'mode should be one of {available_modes}. Received: {mode}')
        self.mode = mode

        available_labels_encoding = ['hot', 'smooth', False]
        if labels_encoding not in available_labels_encoding:
            raise ValueError(f'labels_encoding should be one of {available_labels_encoding}. '
                             f'Received: {labels_encoding}')
        self.labels_encoding = labels_encoding

        if (self.labels_encoding == "smooth") and not (0 < smooth_factor <= 1):
            raise ValueError('Smooth labels encoding must use a smooth_factor '
                             'in the range (0, 1].')

        # Validate augmenter
        # Uncomment and adjust if using albumentations
        # if augmenter and not isinstance(augmenter, Compose):
        #     raise ValueError('augmenter argument must be an instance of albumentations '
        #                      'Compose class. Received type: %s' % type(augmenter))
        self.augmenter = augmenter

        self.src = src
        self.X_key = X_key
        self.y_key = y_key
        self.classes_key = classes_key
        self.shuffle = shuffle if self.mode == 'train' else False
        self.scaler = scaler
        self.num_classes = num_classes
        self.smooth_factor = smooth_factor

        # Handle label scaling
        self.y_scaler = y_scaler
        if self.y_scaler == "minmax":
            self.y_range = y_range
            self.label_scaler = MinMaxScaler(feature_range=y_range)
        elif self.y_scaler == "std":
            self.label_scaler = StandardScaler()
        else:
            self.label_scaler = None

        # Load dataset size
        with h5py.File(self.src, "r") as file:
            self.dataset_length = file[self.X_key].shape[0]

        # Initialize indices
        self._indices = np.arange(self.dataset_length)
        if self.shuffle:
            np.random.shuffle(self._indices)

        # Fit label scaler if needed
        self.fit_label_scaler()

    def __repr__(self):
        """Representation of the class."""
        return f"{self.__class__.__name__}({self.__dict__!r})"

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates one sample of data.

        Arguments
        ---------
        index : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple of torch.Tensors
            A tuple containing the image and its associated label.
        """
        actual_index = self._indices[index]

        with h5py.File(self.src, "r") as file:
            image = file[self.X_key][actual_index]
            label = file[self.y_key][actual_index]

        # Ensure label is a scalar
        if isinstance(label, (list, np.ndarray)):
            label = label[0]  # Adjust based on your label structure

        # Apply data augmentation if in 'train' mode and augmenter is provided
        if self.mode == 'train' and self.augmenter:
            augmented = self.augmenter(image=image)
            image = augmented['image']

        # Apply normalization if required
        if self.scaler:
            image = self.apply_normalization(image)

        # Apply labels encoding
        if self.labels_encoding:
            label = self.apply_labels_encoding(label)

        # Apply y scaling
        if self.y_scaler:
            label = self.apply_label_scaling(label)

        # Convert image and label to PyTorch tensors
        image = torch.from_numpy(image).float()
        if image.dim() == 2:
            image = image.unsqueeze(0)

        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label).float()
        else:
            label = torch.tensor(label, dtype=torch.float32)

        return image, label

    def apply_labels_smoothing(self, label: np.ndarray, factor: float) -> np.ndarray:
        """Applies label smoothing to the original labels binary matrix.

        Arguments
        ---------
        label : np.ndarray
            Current label.
        factor : float
            Smooth factor.

        Returns
        -------
        np.ndarray
            A binary class matrix with smoothed labels.
        """
        if label.ndim == 1:
            # Binary labels: 0 or 1
            label = np.clip(label * (1 - factor) + 0.5 * factor, 0, 1)
        else:
            # Multiclass labels
            label = np.clip(label * (1 - factor) + factor / label.shape[0], 0, 1)
        return label

    def apply_labels_encoding(self, label: Union[int, float, np.ndarray]) -> np.ndarray:
        """Converts a label to binary class matrix if encoding is specified.

        Arguments
        ---------
        label : int or float or np.ndarray
            Current label.

        Returns
        -------
        np.ndarray
            Encoded label.
        """
        if self.num_classes is None:
            raise ValueError("num_classes must be specified for labels encoding.")

        if self.labels_encoding in ["hot", "smooth"]:
            # Convert to one-hot encoding
            label_encoded = np.eye(self.num_classes, dtype='uint8')[int(label)]
            label_encoded = label_encoded.astype(np.float32)
            if self.labels_encoding == "smooth":
                label_encoded = self.apply_labels_smoothing(label_encoded, self.smooth_factor)
            return label_encoded
        else:
            # No encoding
            return label

    def apply_normalization(self, image: np.ndarray) -> np.ndarray:
        """Normalize the pixel intensities.

        Normalize the pixel intensities to the range [-1, 1].

        Arguments
        ---------
        image : np.ndarray
            Image tensor to be normalized.

        Returns
        -------
        np.ndarray
            Normalized image tensor.
        """
        # Ensure data is in float format to handle division properly
        image = image.astype("float32")

        # Calculate min and max values
        min_val = np.min(image)
        max_val = np.max(image)

        # Perform min-max normalization to [0, 1]
        if max_val - min_val == 0:
            image_normalized = image - min_val  # Avoid division by zero
        else:
            image_normalized = (image - min_val) / (max_val - min_val)

        # Scale from [0, 1] to [-1, 1]
        image_normalized = image_normalized * 2 - 1

        return image_normalized

    def apply_label_scaling(self, label: Union[int, float, np.ndarray]) -> float:
        """Applies scaling to the label if y_scaler is set.

        Arguments
        ---------
        label : int or float or np.ndarray
            Current label.

        Returns
        -------
        float
            Scaled label.
        """
        if isinstance(label, (int, float)):
            label = np.array([[label]])
        elif isinstance(label, np.ndarray):
            label = label.reshape(-1, 1)
        else:
            raise ValueError("Unsupported label type for scaling.")

        label_scaled = self.label_scaler.transform(label)
        return label_scaled.flatten()[0]

    def fit_label_scaler(self):
        """Fits the label scaler if y_scaling is enabled."""
        if self.y_scaler and self.label_scaler is not None:
            with h5py.File(self.src, "r") as file:
                labels = np.array(file[self.y_key])
                labels = labels.reshape(-1, 1)
                self.label_scaler.fit(labels)

    def inverse_transform_labels(self, scaled_label: float) -> float:
        """
        Returns the inverse of the transformed label.

        Arguments
        ---------
        scaled_label: float
            The scaled label to be transformed back based on dataset fit.

        Returns
        -------
        float
            The unscaled label value.
        """
        if self.y_scaler and self.label_scaler is not None:
            scaled_label = np.array([[scaled_label]])
            return self.label_scaler.inverse_transform(scaled_label).flatten()[0]
        return scaled_label
