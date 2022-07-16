#!/usr/bin/env python
import yaml

class ExperimentConfig():

    def __init__(self, config):

        self.__config = None
        self.__path = config

        with open(self.__path, "r") as stream:
            try:
                self.__config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def get(self, key, fallback_value=None):

        if key in self.__config:
            return self.__config[key]
        else:
            return fallback_value

    def add(self, args):

        for arg in vars(args):
            print(arg + ' -- ' + str(getattr(args, arg)))
            self.__config[arg] = getattr(args, arg)

    def get_path(self):
        return self.__path
