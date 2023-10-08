from src.data import data_loader

if __name__ == "__main__":
    train_ds, test_ds = data_loader.load_data_gen()

    for x, y in train_ds:
        print()
        print('Training')
        print(x.shape, y.shape)
        print(y[:10])
        break

    for x, y in test_ds:
        print()
        print('Testing')
        print(x.shape, y.shape)
        print(y[:10])
        break
