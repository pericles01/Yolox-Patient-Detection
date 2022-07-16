from __future__ import annotations

import os
from azureml.core import Workspace, Environment, Dataset, Experiment, Datastore, ComputeTarget, ScriptRunConfig, dataset, runconfig, ContainerRegistry
from azureml.data import OutputFileDatasetConfig
from azureml.core.conda_dependencies import CondaDependencies
from azureml.exceptions import UserErrorException
from shutil import copytree, copyfile, rmtree

class AzureAccess:

    def __init__(self, subscription, resource_group) -> None:
        self.__subscription = subscription
        self.__resource_group = resource_group
        self.__ws = None
        self.__ds_data = None
        self.__ds_model = None
        self.__ds_output = None
        self.__snapshot = 'snapshot'

    def workspace(self, workspace_name) -> AzureAccess:
        try:
            self.__ws = Workspace(subscription_id=self.__subscription, resource_group=self.__resource_group, workspace_name=workspace_name)
            self.__ws.write_config()
            print('Library configuration succeeded')
            return self
        except:
            print('ERROR:Workspace not found')
            return None

    def pip_environment(self, env_name, req_path='./requirements.txt') -> AzureAccess:
        self.__env = Environment.from_pip_requirements(env_name, req_path)
        self.__env.register(self.__ws)
        print('Environment created')
        return self

    def dockerfile_environment(self, env_name, dockerfile_name) -> AzureAccess:
        self.__env = Environment.from_dockerfile(env_name, dockerfile_name)
        self.__env.register(self.__ws)
        print('Environment created')
        return self

    def dockerimage_environment(self, env_name, dockerimage_name, registry) -> AzureAccess:
        container_registry = ContainerRegistry()

        self.__env = Environment.from_docker_image(env_name, dockerimage_name, container_registry=registry)
        self.__env.register(self.__ws)
        print('Environment created')
        return self

    def conda_environment(self, env_name, existing_env_name, packages = []) -> AzureAccess:
        env = Environment.get(self.__ws, existing_env_name)
        self.__env = env.clone(env_name)
        conda_dep = CondaDependencies.create(pip_packages=packages)
        self.__env.python.conda_dependencies = conda_dep
        self.__env.register(self.__ws)
        print('Environment created')
        return self

    def input_predefined_dataset(self, name) -> AzureAccess:
        try:
            self.__ds_data = Dataset.get_by_name(self.__ws, name)
            print('Successfully got dataset ' + name)
        except UserErrorException:
            print('ERROR:Dataset not found')
            return None

        return self 

    def input_dataset(self, datastore_name, data_path, model_path) -> AzureAccess:
        try:
            datastore = Datastore.get(self.__ws, datastore_name)

            self.__ds_data = Dataset.File.from_files((datastore, data_path))
            print('Successfully created data input ' + data_path)

            if model_path is not None:
                print('Attempting to load model: ' + model_path)
                self.__ds_model = Dataset.File.from_files((datastore, model_path))
                print('Successfully created model input ' + model_path)

        except UserErrorException:
            print('ERROR:Dataset not found')
            return None

        return self 

    def output_dataset(self, datastore_name, path) -> AzureAccess:
        datastore = Datastore.get(self.__ws, datastore_name)
        self.__ds_output = OutputFileDatasetConfig(destination=(datastore, path))
        print('Successfully created output dataset to ' + path)
        return self

    def compute_target(self, name) -> AzureAccess:
        self.__compute_target = ComputeTarget(workspace=self.__ws, name=name)
        return self

    def experiment(self, name)-> AzureAccess:
        self.__experiment = Experiment(workspace=self.__ws, name=name)
        return self

    def create_snapshot(self, config)->AzureAccess:

        if os.path.exists(self.__snapshot):
            rmtree(self.__snapshot, ignore_errors=True)

        copytree('./src', self.__snapshot)
        copyfile(config, self.__snapshot + '/run_config.yml')

        return self

    def run(self, script_name) -> None:

        input_arguments = []

        if self.__ds_data != None:
            input_arguments.extend(['--indir', self.__ds_data.as_named_input('features').as_mount()])
        if self.__ds_output != None:
            input_arguments.extend(['--outdir', self.__ds_output.as_mount()])
        if self.__ds_model != None:
            input_arguments.extend(['--ckpt', self.__ds_model.as_named_input('model_weigts').as_mount()])

        runconf = runconfig.DockerConfiguration(use_docker=True)

        config = ScriptRunConfig(
            source_directory=self.__snapshot,
            script=script_name,
            arguments=input_arguments,
            compute_target=self.__compute_target,
            environment=self.__env,
            docker_runtime_config=runconf
        )

        run = self.__experiment.submit(config)
        aml_url = run.get_portal_url()

        print("Submitted to computer instance. Click link below")
        print(aml_url)
