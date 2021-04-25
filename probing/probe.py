import logging

import fire

import constants
from experiment import Experiment

logging.basicConfig(format=constants.LOG_FORMAT)
logging.getLogger(constants.LOGGER_NAME).setLevel(logging.INFO)


experiment = Experiment()


if __name__ == "__main__":
    lupa = fire.Fire(Experiment())
