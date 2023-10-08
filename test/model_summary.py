from src.model.vae import VAE
from configs import config

if __name__ == "__main__":
    a = VAE(2)
    a.build(input_shape=config.parameter["input_shape"])
    a.encoder.summary()
    a.decoder.summary()
