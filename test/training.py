from src.model.beta_vae_agent import BetaVAE
from src.data import data_loader
from configs import config

if __name__ == "__main__":
    latent_dim = config.parameter["latent_dim"]
    beta = config.parameter["beta"]
    epochs = config.parameter["epochs"]

    x_train, y_train, _, _ = data_loader.load_data()
    train_ds = data_loader.tensor_slices(x_train)

    agent = BetaVAE("231008")
    agent.train(train_ds, epochs)
    agent.save_model()
