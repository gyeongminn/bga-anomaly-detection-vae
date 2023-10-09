from src.model.vae_agent import VaeAgent
from src.data import data_loader


if __name__ == "__main__":
    x_train, _ = data_loader.load_train_data()

    vae = VaeAgent()
    vae.train(x_train)
    vae.save_model()
