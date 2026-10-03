"""
Convolutional Autoencoder Architecture for Industrial Visual Anomaly Detection.
Built with Keras Functional API.
Compact 4-Level Bottleneck Architecture without Batch Normalization to prevent latent saturation.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple

def build_encoder(input_shape: Tuple[int, int, int] = (256, 256, 3), latent_dim: int = 256) -> Model:
    """
    Builds the 4-level compact Encoder network.
    Input: 256x256x3 -> Bottleneck: 16x16x256.
    """
    inputs = layers.Input(shape=input_shape, name="encoder_input")

    x = layers.Conv2D(32, (3, 3), strides=2, padding="same", name="conv1")(inputs)
    x = layers.LeakyReLU(negative_slope=0.2, name="leaky1")(x)
    x = layers.Conv2D(64, (3, 3), strides=2, padding="same", name="conv2")(x)
    x = layers.LeakyReLU(negative_slope=0.2, name="leaky2")(x)
    x = layers.Conv2D(128, (3, 3), strides=2, padding="same", name="conv3")(x)
    x = layers.LeakyReLU(negative_slope=0.2, name="leaky3")(x)
    x = layers.Conv2D(latent_dim, (3, 3), strides=2, padding="same", name="conv4_bottleneck")(x)
    latent = layers.LeakyReLU(negative_slope=0.2, name="leaky4_latent")(x)

    return Model(inputs, latent, name="encoder")

def build_decoder(latent_shape: Tuple[int, int, int] = (16, 16, 256), output_channels: int = 3) -> Model:
    """
    Builds the 4-level compact Decoder network.
    Latent: 16x16x256 -> Reconstruction: 256x256x3.
    """
    inputs = layers.Input(shape=latent_shape, name="decoder_input")
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, padding="same", name="deconv1")(inputs)
    x = layers.LeakyReLU(negative_slope=0.2, name="leaky_dec1")(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding="same", name="deconv2")(x)
    x = layers.LeakyReLU(negative_slope=0.2, name="leaky_dec2")(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, padding="same", name="deconv3")(x)
    x = layers.LeakyReLU(negative_slope=0.2, name="leaky_dec3")(x)
    outputs = layers.Conv2DTranspose(
        output_channels, (3, 3), strides=2, padding="same", activation="sigmoid", name="reconstruction_output"
    )(x)

    return Model(inputs, outputs, name="decoder")

class ConvAutoencoder(Model):
    """
    Complete Compact Convolutional Autoencoder Model.
    """
    def __init__(self, input_shape: Tuple[int, int, int] = (256, 256, 3), latent_dim: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.input_spec_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder = build_encoder(input_shape, latent_dim)
        self.decoder = build_decoder((16, 16, latent_dim), input_shape[-1])

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        latent = self.encoder(inputs, training=training)
        reconstruction = self.decoder(latent, training=training)
        return reconstruction

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_shape": self.input_spec_shape,
            "latent_dim": self.latent_dim
        })
        return config

def build_conv_autoencoder(input_shape: Tuple[int, int, int] = (256, 256, 3), latent_dim: int = 256) -> ConvAutoencoder:
    """
    Factory function to instantiate compact ConvAutoencoder.
    """
    model = ConvAutoencoder(input_shape=input_shape, latent_dim=latent_dim)
    model.build((None, *input_shape))
    return model
