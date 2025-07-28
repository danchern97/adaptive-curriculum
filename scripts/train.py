import argparse
import yaml
from adaptive_curriculum.data_processing import get_dataset
from adaptive_curriculum.models import get_model


def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset = get_dataset(config['dataset']['name'], **config['dataset']['params'])
    model = get_model(config['model']['name'], **config['model']['params'])

    # TODO: Add training loop
    print(f"Training model {config['model']['name']} on dataset {config['dataset']['name']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args.config) 