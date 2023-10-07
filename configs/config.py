info = {
    "name": "BGA Anomaly detection by exploiting Parallel Variational AutoEncoders",
    "author": "Gyeongmin Lee",
    "email": "gyeongmin.hansung.ac.kr"
}

config = {
    "data_dir": "../data",
    "class_label": ['good', 'extra', 'missing'],
    "image_size": (256, 256),
    "input_shape": (None, 256, 256, 1),
    "random_seed": 123,
    "batch_size": 128,
    "epochs": 100,
    "latent_dim": 2,
    "beta": 1
}