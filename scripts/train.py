import argparse
import yaml
import os
from adaptive_curriculum.data_processing import get_dataset
from adaptive_curriculum.models import get_model

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize wandb if available and enabled
    if WANDB_AVAILABLE and config.get('wandb', {}).get('enabled', True):
        wandb_config = config.get('wandb', {})
        project = wandb_config.get('project', 'adaptive-curriculum')
        run_name = wandb_config.get('run_name', f"train_{config['model']['name']}")
        
        try:
            wandb.init(
                project=project,
                name=run_name,
                config=config
            )
            print(f"Initialized wandb run: {wandb.run.name}")
        except Exception as e:
            print(f"Failed to initialize wandb: {e}")

    dataset = get_dataset(config['dataset']['name'], **config['dataset']['params'])
    model = get_model(config['model']['name'], **config['model']['params'])

    print(f"Training model {config['model']['name']} on dataset {config['dataset']['name']}")
    
    # TODO: Add actual training loop here
    # This would include:
    # - Training iterations
    # - Loss calculation
    # - Logging metrics to wandb
    
    # Example wandb logging (when actual training is implemented):
    # if WANDB_AVAILABLE and wandb.run is not None:
    #     wandb.log({"epoch": epoch, "loss": loss, "accuracy": accuracy})
    
    # Finish wandb run
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()
        print("Wandb run finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    main(args.config) 