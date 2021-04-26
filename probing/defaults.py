from typing import List
from constants import BERT_MULTILINGUAL_A_NAME, SUBSAMPLES_NAMES


class SubsamplesDataEntry:
    SUBSAMPLES_NAMES = SUBSAMPLES_NAMES

    def __init__(self, train=None, test=None, dev=None):
        self.train = train
        self.test = test
        self.dev = dev

        for subsample_name in self.SUBSAMPLES_NAMES:
            try:
                self[subsample_name]
            except Exception as e:
                raise e

    def __getitem__(self, item):
        assert item in self.SUBSAMPLES_NAMES
        try:
            return getattr(self, item)
        except Exception as e:
            raise KeyError(f"config for subsample {e} not set", e)


DEFAULT_MODEL_ARCHITECTURE: str = BERT_MULTILINGUAL_A_NAME
DEFAULT_WHETHER_TO_REINITIALIZE_MODEL_W_RANDOM_WEIGHTS_OR_NOT: bool = False
DEFAULT_WHETHER_TO_USE_FINETUNED_MODEL_OR_NOT: bool = False
DEFAULT_LAYERS_TO_PROBE: List[int] = [11, ]
DEFAULT_DEV_PROBING_NUMBER_OF_SENTENCES: int = 0
DEFAULT_TRAIN_PROBING_NUMBER_OF_SENTENCES: int = 1500
DEFAULT_TEST_PROBING_NUMBER_OF_SENTENCES: int = 1000
DEFAULT_FINETUNED_MODEL_PATH = "finetuned_model/"
