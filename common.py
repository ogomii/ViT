class TrainingConfig:
    def __init__(self, 
                n_epochs=5, 
                batch_size=16, 
                learning_rate=0.01, 
                loss_fn=None,
                patience=10,
                early_stop=True,
                weight_decay=1e-3,
                model_save_path='model.pth'
                ):
        self.n_epochs: int = n_epochs
        self.batch_size: int = batch_size
        self.learning_rate: float = learning_rate
        self.loss_fn = loss_fn 
        self.patience: int = patience
        self.early_stop: bool = early_stop
        self.weight_decay: float = weight_decay
        self.model_save_path: str = model_save_path

    def __str__(self):
        return f'''TrainingConfig(
                n_epochs={self.n_epochs}, 
                batch_size={self.batch_size}, 
                learning_rate={self.learning_rate}, 
                loss_fn={self.loss_fn}
                patience={self.patience},
                early_stop={self.early_stop},
                weight_decay={self.weight_decay},
                model_save_path={self.model_save_path}
                )'''
