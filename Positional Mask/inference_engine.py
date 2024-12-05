#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------


from pathlib import Path
from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

from health_multimodal.image.data.io import load_image
from health_multimodal.image.data.transforms import infer_resize_params
from health_multimodal.image.model.model import ImageModel

TypeShape2D = Tuple[int, int]


class ImageInferenceEngine:
    """
    Encapsulate inference-time operations on an image model.
    """

    def __init__(self, image_model: ImageModel, transform: Compose):
        """
        :param img_model: Trained image model
        :param transform: Transform to apply to the image after loading. Must return a torch.Tensor that can be
            input directly to the image model.
        """

        assert isinstance(image_model, ImageModel), f"Expected an ImageModel, got {type(image_model)}"

        self.model = image_model
        self.transform = transform

        self.model.eval()
        self.resize_size, self.crop_size = infer_resize_params(self.transform.transforms)
        self.to = self.model.to

    def load_and_transform_input_image(self, image_path: Path, transform: Callable) -> Tuple[torch.Tensor, TypeShape2D]:
        """Read an image and apply the transform to it.

        1. Read the image from the given path
        2. Apply transform
        3. Add the batch dimension
        4. Move to the correct device

        :param return_original_shape: Whether to return an extra tuple that has the original shape of the image
            before the transforms. The tuple returned contains (width, height).
        """

        image = load_image(image_path)
        device = next(self.model.parameters()).device
        transformed_image = transform(image).unsqueeze(0).to(device)
        return transformed_image, image.size

    
    def load_and_transform_input_image1(self, image_path: Path, transform: Callable, position: str) -> Tuple[torch.Tensor, TypeShape2D]:
        
        """
        -----------------------------------------------------------------------------------------------------------------
        Read an image, apply the transform to it, and mask a specific part of the image.
    
        1. Read the image from the given path
        2. Apply transform
        3. Mask a specified part of the image (right, left, top, or bottom)
        4. Add the batch dimension
        5. Move to the correct device
    
        :param mask_part: Specifies which part of the image to mask ('right', 'left', 'top', 'bottom')
        :return: Transformed image tensor and the original image size.
        """

        #"""
        image = load_image(image_path)
        device = next(self.model.parameters()).device
        transformed_image = transform(image).unsqueeze(0).to(device)
        
        # Get the height and width of the transformed image
        _, _, height, width = transformed_image.shape
    
        # Apply masking based on the specified part
        if position == 'right':
            transformed_image[:, :, :, width // 2:] = 0  # Mask the right half
        elif position == 'left':
            transformed_image[:, :, :, :width // 2] = 0  # Mask the left half
        elif position == 'bottom':
            transformed_image[:, :, :height // 2, :] = 0  # Mask the top half
        elif position == 'top':
            transformed_image[:, :, height // 2:, :] = 0  # Mask the bottom half
        elif position == 'right top':
            transformed_image[:, :, height // 2:, :] = 0  # Mask the bottom half
            transformed_image[:, :, :, width // 2:] = 0  # Mask the right half
        elif position == 'right bottom':
            transformed_image[:, :, :height // 2, :] = 0  # Mask the top half
            transformed_image[:, :, :, width // 2:] = 0  # Mask the right half
        elif position == 'left top':
            transformed_image[:, :, height // 2:, :] = 0  # Mask the bottom half
            transformed_image[:, :, :, :width // 2] = 0  # Mask the left half
        elif position == 'left bottom':
            transformed_image[:, :, :height // 2, :] = 0  # Mask the top half
            transformed_image[:, :, :, :width // 2] = 0  # Mask the left half
    
        return transformed_image, image.size


    @torch.no_grad()
    def get_projected_patch_embeddings(self, image_path: Path) -> Tuple[torch.Tensor, TypeShape2D]:
        """Compute image patch embeddings in the joint latent space, preserving the image grid.

        :param image_path: Path to the image to compute embeddings for.
        :return: A tuple containing the image patch embeddings and
            the shape of the original image (width, height) before applying transforms.
        """
        input_image, img_shape = self.load_and_transform_input_image(image_path, self.transform)
        projected_img_emb = self.model.get_patchwise_projected_embeddings(input_image, normalize=True)
        assert projected_img_emb.shape[0] == 1

        return projected_img_emb[0], img_shape

    @torch.no_grad()
    def get_projected_global_embedding(self, image_path: Path) -> torch.Tensor:
        """Compute global image embedding in the joint latent space.

        :param image_path: Path to the image to compute embeddings for.
        :return: Torch tensor containing l2-normalised global image embedding [joint_feature_dim,]
                 where joint_feature_dim is the dimensionality of the joint latent space.
        """
        input_image, _ = self.load_and_transform_input_image(image_path, self.transform)
        projected_img_emb = self.model.forward(input_image).projected_global_embedding
        projected_img_emb = F.normalize(projected_img_emb, dim=-1)

        assert projected_img_emb.shape[0] == 1
        assert projected_img_emb.ndim == 2

        return projected_img_emb[0]

    @torch.no_grad()
    def get_projected_global_embedding1(self, image_path: Path, position: str) -> torch.Tensor:
        """Compute global image embedding in the joint latent space.

        :param image_path: Path to the image to compute embeddings for.
        :return: Torch tensor containing l2-normalised global image embedding [joint_feature_dim,]
                 where joint_feature_dim is the dimensionality of the joint latent space.
        """
        input_image, _ = self.load_and_transform_input_image1(image_path, self.transform, position)
        projected_img_emb = self.model.forward(input_image).projected_global_embedding
        projected_img_emb = F.normalize(projected_img_emb, dim=-1)

        assert projected_img_emb.shape[0] == 1
        assert projected_img_emb.ndim == 2

        return projected_img_emb[0]
