import logging

import fire

import constants
from experiment import Experiment

logging.basicConfig(format=constants.LOG_FORMAT)
logging.getLogger(constants.LOGGER_NAME).setLevel(logging.INFO)

# the func below is a wrapped because of some python-fire docstrings issues

experiment = Experiment()


if __name__ == "__main__":
    fire_experiment = fire.Fire(Experiment())
