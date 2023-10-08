from src.model.beta_vae import BetaVAE
from configs import config

if __name__ == "__main__":
    a = BetaVAE(2)
    a.build(input_shape=config.parameter['input_shape'])
    a.encoder.summary()
    a.decoder.summary()
