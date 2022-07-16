from os import pipe
import argparse
from src.experiment_config import ExperimentConfig
from src.azure_access import AzureAccess


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train the YoloX model with different complexities.')
    parser.add_argument('target', type=str, help='name of compute instance to run the code')
    parser.add_argument('-config', type=str, help="path to config file, see samples in folder configs/")
    args = parser.parse_args()

    print('Using config ' + args.config)

    conf = ExperimentConfig(args.config)

    if args.target == 'local':
        from src.infer import infer
        infer(conf, 'src/')
    else:
        AzureAccess(conf.get('subscription'), conf.get('resource_group')) \
            .workspace(conf.get('workspace')) \
            .dockerimage_environment(conf.get('environment'), conf.get('docker_image'), conf.get('container_registry')) \
            .input_dataset(conf.get('datastore'), conf.get('indir'), conf.get('ckpt')) \
            .output_dataset(conf.get('datastore'), conf.get('outdir')) \
            .compute_target(args.target) \
            .experiment(conf.get('experiment')) \
            .create_snapshot(args.config) \
            .run('infer.py')
