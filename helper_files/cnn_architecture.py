#!/bin/python3.8

import torch
import torch.nn as nn

from src.sim.ros.python3_ros_ws.src.handcrafted_cone_detection.helper_files.local_base_net import Local_Base_Net,ArchitectureConfig
from src.ai.utils import mlp_creator
from src.core.data_types import Action
from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension

"""
Tiny four encoding and three decoding layers with dropout.
Expects 1*200*200 inputs and outputs 6c 
"""


class Net(Local_Base_Net):

    def __init__(self, config: ArchitectureConfig, quiet: bool = False):
        super().__init__(config=config, quiet=True)
        self._logger = get_logger(name=get_filename_without_extension(__file__),
                                  output_path=config.output_path,
                                  quiet=False)
        if not quiet:
            cprint(f'Started.', self._logger)
        self.input_size = (1, 200, 212)
        self.output_size = (6,)
        self.discrete = False
        self.dropout = nn.Dropout(p=config.dropout) if config.dropout != 'default' else None
        self.batch_normalisation = config.batch_normalisation if isinstance(config.batch_normalisation, bool) \
            else False
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 12, 3, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(12, 24, 3, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(24, 24, 3, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(24, 64, 3, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=2),
            nn.ReLU(),
        )
        self.decoder = mlp_creator(sizes=[576, 64, self.output_size[0]],
                                   activation=nn.ReLU(),
                                   output_activation=nn.Identity(),
                                   bias_in_last_layer=False)
        #dont need init, we use checkpoint
        #self.initialize_architecture()


    def forward(self, inputs, train: bool = False) -> torch.Tensor:
        """
        Outputs steering action only
        """
        self.set_mode(train)
        inputs = self.process_inputs(inputs=inputs)
        if self._config.finetune:
            with torch.no_grad():
                x = self.encoder(inputs)
        else:
            x = self.encoder(inputs)
        x = x.flatten(start_dim=1)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.decoder(x)
        return x

    def get_action(self, inputs, train: bool = False) -> Action:
        inputs = self.process_inputs(inputs=inputs)
        output = self.forward(inputs)
        return Action(actor_name=get_filename_without_extension(__file__),
                      value=output.data)

