import torch
import numpy as np
from cv2 import cv2
from typing import Union
import torch.nn as nn
"""
BaseClass for neural network architectures.
Include functionality to initialize network, load and store checkpoints according to config.
All other architectures in src.ai.architectures should inherit from this class.
"""

class ArchitectureConfig():
    architecture: str = 'cnn_architecture' # name of architecture to be loaded
    initialisation_type: str = 'xavier'
    random_seed: int = 123
    device: str = 'cpu'
    latent_dim: str = 'default'
    vae: str = 'default'
    finetune: bool = False
    dropout: float = 0.5
    batch_normalisation: str = 'default'
    dtype: str = 'default'
    log_std: str = 'default'
    output_path = '/media/thomas/Elements/training_nn/jupiter_test_path'

class Local_Base_Net(nn.Module):

    def __init__(self, config: ArchitectureConfig, quiet: bool = True):
        super().__init__()
        # input range from 0 -> 1 is expected, for range -1 -> 1 this field should state 'zero_centered'
        self.input_scope = 'default'

        self.input_size = None
        self.output_size = None
        self.discrete = None
        self._config = config
        self.dtype = torch.float32 if config.dtype == 'default' else eval(f"torch.{config.dtype}")
        self._device = torch.device(
            "cuda" if self._config.device in ['gpu', 'cuda'] and torch.cuda.is_available() else "cpu"
        )
        self.global_step = torch.as_tensor(0, dtype=torch.int32)

    def initialize_architecture(self):
        torch.manual_seed(self._config.random_seed)
        torch.set_num_threads(1)

    def set_device(self, device: Union[str, torch.device]):
        self._device = torch.device(
            "cuda" if device in ['gpu', 'cuda'] and torch.cuda.is_available() else "cpu"
        ) if isinstance(device, str) else device
        try:
            for layer in self.modules():
                layer.to(self._device)
        except AssertionError:
            print(f'failed to work on {self._device} so working on cpu')
            self._device = torch.device('cpu')
            self.to(self._device)

    def set_mode(self, train: bool = False):
        self.train(train)

    def process_inputs(self, inputs: Union[torch.Tensor, np.ndarray, list, int, float]) -> torch.Tensor:
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        if len(self.input_size) == 3:
            # check if 2D input is correct
            # compare channel first / last for single image:
            if len(inputs.shape) == 3 and inputs.shape[-1] == self.input_size[0]:
                # in case it's channel last, assume single raw data input which requires preprocess:
                # check for size
                if inputs.shape[1] != self.input_size[1]:
                    # resize with opencv
                    inputs = cv2.resize(np.asarray(inputs), dsize=(self.input_size[1], self.input_size[2]),
                                        interpolation=cv2.INTER_LANCZOS4)
                    if self.input_size[0] == 1:
                        inputs = inputs.mean(axis=-1, keepdims=True)
                # check for scope
                if inputs.max() > 1 or inputs.min() < 0:
                    inputs += inputs.min()
                    inputs = torch.as_tensor(inputs, dtype=torch.float32)
                    inputs /= inputs.max()
                if self.input_scope == 'zero_centered':
                    inputs *= 2
                    inputs -= 1
                # make channel first and add batch dimension
                inputs = torch.as_tensor(inputs).permute(2, 0, 1).unsqueeze(0)

        # create Tensors
        if not isinstance(inputs, torch.Tensor):
            try:
                inputs = torch.as_tensor(inputs, dtype=self.dtype)
            except ValueError:
                inputs = torch.stack(inputs).type(self.dtype)
        inputs = inputs.type(self.dtype)

        # add batch dimension if required
        if len(self.input_size) == len(inputs.size()):
            inputs = inputs.unsqueeze(0)
        # put inputs on device
        inputs = inputs.to(self._device)

        return inputs

    def get_device(self) -> torch.device:
        return self._device

    def count_parameters(self) -> int:
        count = 0
        for p in self.parameters():
            count += np.prod(p.shape)
        return count

    def remove(self):
        [h.close() for h in self._logger.handlers]

    def get_checkpoint(self) -> dict:
        """
        :return: a dictionary with global_step and model_state of neural network.
        """
        return {
            'global_step': self.global_step,
            'model_state': self.state_dict()
        }
    def load_checkpoint(self, checkpoint) -> None:
        """
        Try to load checkpoint in global step and model state. Raise error.
        :param checkpoint: dictionary containing 'global step' and 'model state'
        :return: None
        """
        self.global_step = checkpoint['global_step']
        self.load_state_dict(checkpoint['model_state'])
        self.set_device(self._device)
        #print(f'checksum: {self.get_checksum()}')