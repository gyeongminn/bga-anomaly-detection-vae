import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

info = {
    "name": "BGA Anomaly Detection",
    "author": "Gyeongmin Lee",
    "email": "gyeongmin.hansung.ac.kr",
}

parameter = {
    "base_dir": "D:/work/Github/bga-anomaly-detection-vae",
    "dataset": "data1",
    "data1_class_names": ["good", "extra", "missing", "pitch", "size"],
    "data2_class_names": ["good", "anomaly"],
    "image_size": (256, 256),
    "input_shape": (None, 256, 256, 1),
    "random_seed": 123,
    "batch_size": 128,
    "epochs": 1000,
    "latent_dim": 2,
    "learning_rate": 1e-3,
}

if parameter['dataset'] == "data1":
    parameter["class_names"] = parameter["data1_class_names"]
elif parameter['dataset'] == "data2":
    parameter["class_names"] = parameter["data2_class_names"]
