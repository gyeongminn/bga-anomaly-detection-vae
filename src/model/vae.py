import tensorflow as tf
from keras.models import Model

from src.model.encoder import Encoder
from src.model.decoder import Decoder


class VAE(Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder()

    @staticmethod
    def reparameterize(mu, log_var):
        epsilon = tf.random.normal(shape=tf.shape(mu))
        return epsilon * tf.exp(0.5 * log_var) + mu

    def call(self, x):
        mu, log_var = self.encoder(x)
        encoded = self.reparameterize(mu, log_var)
        decoded = self.decoder(encoded)
        return decoded, encoded, mu, log_var
