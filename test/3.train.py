from src.model.parallel_vaes import ParallelVAEs
from src.data import data_loader


if __name__ == "__main__":
    x_train, _ = data_loader.load_train_data()

    pvae = ParallelVAEs("231009")
    pvae.train(x_train)
    pvae.save_model()
