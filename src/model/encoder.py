from keras.layers import Conv2D, Dense, Flatten
from keras.models import Model


class Encoder(Model):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.h1 = Conv2D(16, 3, strides=2, padding="same", activation="relu")
        self.h2 = Conv2D(32, 3, strides=2, padding="same", activation="relu")
        self.h3 = Dense(16, activation="relu")
        self.flatten = Flatten()
        self.mu = Dense(latent_dim)
        self.log_var = Dense(latent_dim)

    def call(self, x, training=None, mask=None):
        x = self.h1(x)
        x = self.h2(x)
        x = self.h3(x)
        x = self.flatten(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var
