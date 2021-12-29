from pathlib import Path

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint, EarlyStopping


def train(system_class,
          config,
          experiment_dir="experiments",
          checkpoint_name="vae_{val_loss:.3f}.ckpt"):
    seed_everything(config["random_seed"])
    system = system_class(config)

    experiment_name = config["name"]
    experiment_dir = Path(f"{experiment_dir}/{experiment_name}")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=experiment_dir,
        filename=checkpoint_name,
        auto_insert_metric_name=True
    )

    trainer = Trainer(
        **config["trainer"],
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor="val_loss", patience=3),
            TQDMProgressBar(refresh_rate=20)
        ],
        logger=None
    )
    trainer.fit(system)
    return system, trainer
