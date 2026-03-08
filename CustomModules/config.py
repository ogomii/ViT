class BaseViTConfig:
    """Base configuration class for Vision Transformer models."""
    
    def __init__(self, 
                _channels=3,
                _height=32, 
                _width=32,
                _n_patches=4,
                _d_model=1024,
                _n_heads=16,
                _n_layers=24,
                _dropout_rate=0.2):
        self.channels = _channels
        self.height = _height
        self.width = _width
        self.n_patches = _n_patches  # number of patches in one dimension, so total patches = n_patches^2
        self.patch_size = int(_height / _n_patches)
        self.d_model = _d_model
        self.n_heads = _n_heads
        self.n_layers = _n_layers
        self.dropout_rate = _dropout_rate
    
    def to_dict(self):
        return {
            'channels': self.channels,
            'height': self.height,
            'width': self.width,
            'n_patches': self.n_patches,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'dropout_rate': self.dropout_rate
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(
            _channels=config_dict['channels'],
            _height=config_dict['height'],
            _width=config_dict['width'],
            _n_patches=config_dict['n_patches'],
            _d_model=config_dict['d_model'],
            _n_heads=config_dict['n_heads'],
            _n_layers=config_dict['n_layers'],
            _dropout_rate=config_dict['dropout_rate']
        )


class ViT_config(BaseViTConfig):
    """Configuration for Vision Transformer model."""
    pass


class DeformableViT_config(BaseViTConfig):
    """Configuration for Deformable Vision Transformer model with window-based attention."""
    
    def __init__(self,
                _channels=3,
                _height=32,
                _width=32,
                _n_patches=4,
                _window_size=2,
                _d_model=1024,
                _n_heads=16,
                _n_layers=24,
                _dropout_rate=0.2):
        super().__init__(_channels, _height, _width, _n_patches, _d_model, _n_heads, _n_layers, _dropout_rate)
        self.window_size = _window_size  # size of local attention window
    
    def to_dict(self):
        config_dict = super().to_dict()
        config_dict['window_size'] = self.window_size
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(
            _channels=config_dict['channels'],
            _height=config_dict['height'],
            _width=config_dict['width'],
            _n_patches=config_dict['n_patches'],
            _window_size=config_dict['window_size'],
            _d_model=config_dict['d_model'],
            _n_heads=config_dict['n_heads'],
            _n_layers=config_dict['n_layers'],
            _dropout_rate=config_dict['dropout_rate']
        )
