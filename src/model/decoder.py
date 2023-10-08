from keras.layers import Conv2DTranspose, Dense, Reshape
from keras.models import Model


class Decoder(Model):

    def __init__(self):
        super(Decoder, self).__init__()

        self.fc = Dense(64 * 64 * 64, activation='relu')
        self.reshape = Reshape((64, 64, 64))
        self.h1 = Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')
        self.h2 = Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')
        self.output_layer = Conv2DTranspose(1, 3, padding='same', activation='sigmoid')

    def call(self, z):
        x = self.fc(z)
        x = self.reshape(x)
        x = self.h1(x)
        x = self.h2(x)
        x = self.output_layer(x)

        return x
