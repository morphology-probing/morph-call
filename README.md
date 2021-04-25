## Morph Cal


## usage

```bash
me@my-laptop:~$ python3 probe.py --help
INFO: Showing help with the command 'probe.py -- --help'.

NAME
    probe.py - configure the experiment AND perform probing

SYNOPSIS
    probe.py <flags>

DESCRIPTION
    configure the experiment AND perform probing

FLAGS
    --ud_treebanks_path=UD_TREEBANKS_PATH
        Type: str
        Default: '.'
        path to folder with CONLLU-formatted treebanks to use for probing
    --results_path=RESULTS_PATH
        Type: Optional[str]
        Default: None
        path to folder to store probing results and model inference activations
    --model_architecture=MODEL_ARCHITECTURE
        Type: typ...
        Default: 'bert multilingual'
        one of ["bert multilingual"] ("XLMR" support is coming soon)
    --model_is_finetuned=MODEL_IS_FINETUNED
        Type: bool
        Default: False
        whether to perform experiment on finetuned model or not
    --model_finetuned_path=MODEL_FINETUNED_PATH
        Type: Optional[str]
        Default: None
        (only if model_is_finetuned is True) path to folder where the finetuned model files are stored. defaults to "finetuned_model/"
    --model_finetuned_config_google_url=MODEL_FINETUNED_CONFIG_GOOGLE_URL
        Type: Optional[]
        Default: None
        (only if model_is_finetuned is True) url of finetuned model config.json if the config needs to be downloaded from the google drive
    --model_finetuned_model_google_url=MODEL_FINETUNED_MODEL_GOOGLE_URL
        Type: Optional[]
        Default: None
        (only if model_is_finetuned is True) url of finetuned model bin file if the model needs to be downloaded from the google drive
    --model_is_random=MODEL_IS_RANDOM
        Type: bool
        Default: False
        whether to reinitialize weights of model with random numbers or not
    --layers_to_probe=LAYERS_TO_PROBE
        Type: List
        Default: 'all'
        (either "all" or list w. possible numbers from 0 to 11) -- model layers to perform activations probing onto. example values: [1, 3, 11] or [0, 1, 2] or "all"
    --train_n_sentences=TRAIN_N_SENTENCES
        Type: int
        Default: 1500
        number of sentences to take from probing dataset to train the probing classifier
    --test_n_sentences=TEST_N_SENTENCES
        Type: int
        Default: 1000
        number of sentences to take from probing dataset to evaluate the probing classifier
    --dev_n_sentences=DEV_N_SENTENCES
        Type: int
        Default: 0
        DEPRECATED
```
