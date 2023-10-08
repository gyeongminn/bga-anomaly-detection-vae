import tensorflow as tf
from keras.models import Model

from src.model.vae import VAE
from src.visualization.plots import show_history
from configs import config


class BetaVAE(Model):
    def __init__(self, model_name: str):
        super(BetaVAE, self).__init__()

        self.model_name = model_name
        self.latent_dim = config.parameter["latent_dim"]
        self.beta = config.parameter["beta"]
        self.beta_vae = VAE(config.parameter["latent_dim"])
        self.beta_vae.build(input_shape=config.parameter["input_shape"])
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.parameter["learning_rate"]
        )

    def loss_function(self, x):
        x_hat, z, mu, log_var = self.beta_vae(x)
        recons_loss = tf.reduce_mean(
            tf.reduce_sum(
                # tf.keras.losses.binary_crossentropy(x, x_hat),
                tf.keras.losses.MSE(x, x_hat),
                axis=(1, 2),
            )
        )

        kl_loss = -0.5 * (1 + log_var - tf.square(mu) - tf.exp(log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        loss = recons_loss + self.beta * kl_loss
        return loss, recons_loss, kl_loss

    def train_step(self, x):
        with tf.GradientTape() as tape:
            loss, recons_loss, kl_loss = self.loss_function(x)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"total_loss": loss, "recons_loss": recons_loss, "kl_loss": kl_loss}

    def save_model(self):
        model_dir = config.parameter["base_dir"] + "/weights/" + self.model_name + ".h5"
        try:
            self.beta_vae.save_weights(model_dir)
            print("model saved")
        except (Exception,):
            print(f"model {self.model_name} saved")
            exit(1)

    def load_model(self):
        model_dir = config.parameter["base_dir"] + "/weights/" + self.model_name + ".h5"
        try:
            self.beta_vae.load_weights(model_dir)
            print(f"model {self.model_name} loaded")
        except (Exception,):
            print("weights file not found")
            exit(1)

    def train(self, train_dataset, epochs):
        train_loss_history = []
        kl_loss_history = []
        recons_loss_history = []

        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")

            epoch_train_loss = []
            epoch_kl_loss = []
            epoch_recons_loss = []

            for step, batch in enumerate(train_dataset):
                loss_dict = self.train_step(batch)
                epoch_train_loss.append(loss_dict["total_loss"].numpy())
                epoch_recons_loss.append(loss_dict["recons_loss"].numpy())
                epoch_kl_loss.append(loss_dict["kl_loss"].numpy())

                if step % 100 == 0:
                    loss = loss_dict["total_loss"]
                    recons_loss = loss_dict["recons_loss"]
                    kl_loss = loss_dict["kl_loss"]
                    print(
                        f"Total Loss: {loss:.4f}, Recon Loss: {recons_loss:.4f}, KL Loss: {kl_loss:.4f}"
                    )

            train_loss_history.append(tf.reduce_mean(epoch_train_loss))
            kl_loss_history.append(tf.reduce_mean(epoch_kl_loss))
            recons_loss_history.append(tf.reduce_mean(epoch_recons_loss))

        show_history(train_loss_history, kl_loss_history, recons_loss_history)
