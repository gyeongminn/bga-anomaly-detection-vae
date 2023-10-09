from src.model.shallow.shallow_vae import ShallowVAE
from src.model.deep.deep_vae import DeepVAE
from src.data.data_loader import tensor_slices
from configs import config


class ParallelVAEs:
    def __init__(self, model_name: str):
        self.shallow_vae = ShallowVAE(model_name)
        self.deep_vae = DeepVAE(model_name)

    def get_shallow_vae(self):
        return self.shallow_vae

    def get_deep_vae(self):
        return self.shallow_vae

    def train(self, x):
        epochs = config.parameter["epochs"]
        dataset = tensor_slices(x)
        self.shallow_vae.train(dataset, epochs)
        self.deep_vae.train(dataset, epochs)

    def train_shallow(self, x):
        epochs = config.parameter["epochs"]
        dataset = tensor_slices(x)
        self.shallow_vae.train(dataset, epochs)
        self.shallow_vae.save_model()

    def train_deep(self, x):
        epochs = config.parameter["epochs"]
        dataset = tensor_slices(x)
        self.deep_vae.train(dataset, epochs)
        self.deep_vae.save_model()

    def predict(self, x):
        dataset = tensor_slices(x)
        data_shallow = self.shallow_vae.vae.predict(dataset)[0]
        data_deep = self.deep_vae.vae.predict(dataset)[0]
        return data_shallow, data_deep

    def save_model(self):
        self.shallow_vae.save_model()
        self.deep_vae.save_model()

    def load_model(self):
        self.shallow_vae.load_model()
        self.deep_vae.load_model()
