# Adaptive Curriculum

This project is designed for flexible and extensible machine learning experimentation.

## Project Structure

- `configs/`: Contains configuration files for experiments.
- `data_processing/`: All data loading and processing logic resides here.
- `models/`: Model definitions are stored in this directory.
- `scripts/`: Contains training and evaluation scripts.

## How to Add a New Dataset

1.  Create a new Python file in the `data_processing/` directory (e.g., `my_dataset.py`).
2.  Implement your dataset class in this file.
3.  Register your dataset using the `@register_dataset` decorator.

```python
from . import register_dataset

@register_dataset('my_awesome_dataset')
class MyAwesomeDataset:
    def __init__(self, path):
        self.path = path
        # ... your dataset loading logic ...
```

4.  In your config file, you can now use `my_awesome_dataset`.

## How to Add a New Model

1.  Create a new Python file in the `models/` directory (e.g., `my_model.py`).
2.  Implement your model class in this file.
3.  Register your model using the `@register_model` decorator.

```python
from . import register_model

@register_model('my_awesome_model')
class MyAwesomeModel:
    def __init__(self, input_size, output_size):
        # ... your model definition ...
```

4.  In your config file, you can now use `my_awesome_model`.

## Running the Training

To run the training, use the `train.py` script and provide a configuration file:

```bash
python scripts/train.py --config configs/example_config.yaml
```
