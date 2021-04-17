#!/bin/python3.8
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