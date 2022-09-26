import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import utils
import os


def normalize_image(batch):
    batch = batch / 255.
    return batch


def create_encoder(input_shape, latent_dim):
    encoder_inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(64, 3, strides=2, padding="same")(encoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(256, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(2048)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="Encoder")
    encoder.summary()

    return encoder


def create_decoder(latent_dim):
    latent_inputs = keras.Input(shape=(latent_dim,))

    x = layers.Dense(2048)(latent_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Dense(16384)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Reshape((8,8,256))(x)

    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2DTranspose(16, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="Decoder")
    decoder.summary()

    return decoder


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss           
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def build_model(save_model: bool = False):
    """
    Fit the vae model

    Args:
        save_model (bool, optional): Save the encoder and decoder. Defaults to False.
    """    

    image_size = utils.image_size
    latent_dim = utils.latent_dim
    batch_size = utils.batch_size
    n_epochs= utils.n_epochs
    input_shape = (image_size, image_size, 3)

    data_path = os.path.abspath(os.path.join(__file__, utils._root_data_path, utils._full_data_directory))

    # if the data directory doesn't exist, run the function to create it and join the data
    if not os.path.exists(data_path):
        utils.join_data()

    # Generates a tf.data.Dataset from image files in a directory.
    batch_dataset = tf.keras.utils.image_dataset_from_directory(
        data_path,
        # since vae it is a unsupervised model, we don't need the labels
        label_mode = None,
        seed=42,
        image_size=(image_size, image_size),
        batch_size=batch_size
    )

    # normalized dataset
    batch_dataset_norm = batch_dataset.map(normalize_image)

    encoder = create_encoder(input_shape, latent_dim)
    decoder = create_decoder(latent_dim)
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(1e-4))
    vae.fit(batch_dataset_norm, epochs=n_epochs)

    if save_model:
        vae.decoder.save('model_keras_example_decoder')
        vae.encoder.save('model_keras_example_encoder')

    return vae


if __name__ == '__main__':
    build_model(save_model=True)