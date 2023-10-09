import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

info = {
    "name": "BGA Anomaly Detection",
    "author": "Gyeongmin Lee",
    "email": "gyeongmin.hansung.ac.kr",
}

parameter = {
    "base_dir": "D:/work/Github/bga-anomaly-detection",
    "class_names": ["good", "extra", "missing", "pitch", "size"],
    "image_size": (256, 256),
    "input_shape": (None, 256, 256, 1),
    "random_seed": 123,
    "batch_size": 128,
    "epochs": 1000,
    "latent_dim": 2,
    "learning_rate": 1e-3,
}
