from src.data import data_loader

if __name__ == "__main__":
    x_train, y_train = data_loader.load_train_data()
    print(x_train.shape)
    print(y_train.shape)

    x_test, y_test = data_loader.load_test_data()
    print(x_test.shape)
    print(y_test.shape)
