from src.azure_access import AzureAccess
from src.experiment_config import ExperimentConfig
import argparse
import sys


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train the YoloX model with different complexities.')
    parser.add_argument('target', type=str, help='name of compute instance to run the code')
    parser.add_argument('-config', default='configs/train_default.yml', type=str, help="path to config file, see samples in folder configs/")
    args = parser.parse_args()

    print('Using config ' + args.config)

    # Note: args.config will be copied to the snapshot folder and later used by the train.py script.
    conf = ExperimentConfig(args.config)

    if args.target == 'local':
        sys.path.append('./src') # Needed to load more modules in train.py
        from train import train
        train(conf, 'src/')
    else:
        AzureAccess(conf.get('subscription'), conf.get('resource_group')) \
            .workspace(conf.get('workspace')) \
            .dockerimage_environment(conf.get('environment'), conf.get('docker_image'), conf.get('container_registry')) \
            .input_dataset(conf.get('datastore'), conf.get('indir'), conf.get('ckpt')) \
            .output_dataset(conf.get('datastore'), conf.get('outdir')) \
            .compute_target(args.target) \
            .experiment(conf.get('experiment')) \
            .create_snapshot(args.config) \
            .run('train.py')
