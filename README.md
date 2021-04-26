## Morph Call
Morph Call is a suite of 46 probing tasks for four Indo-European languages that fall under different morphology: Russian, French, English, and German. The tasks are designed to explore the morphosyntactic content of multilingual transformers which is a less studied aspect at the moment.

The tasks are divided into four groups:

* [Morphosyntactic Features](https://github.com/morphology-probing/morph-call/tree/main/data/morphosyntactic_features): probe the encoder for the occurrence of the morphosyntactic properties.
* Masked Token: analogous to [Morphosyntactic Features](https://github.com/morphology-probing/morph-call/tree/main/data/morphosyntactic_features) with the exception that the target word is replaced with a tokenizer-specific mask token.
* [Morphosyntactic Values](https://github.com/morphology-probing/morph-call/tree/main/data/morphosyntactic_values): is a group of k-way classification tasks for each feature where *k* is the number of values that the feature can take.
* [Perturbations](https://github.com/morphology-probing/morph-call/tree/main/data/perturbations): tasks test the encoder sensitivity to syntactic and inflectional sentence perturbations.

## Probing Methods

* [Supervised probing](https://github.com/morphology-probing/morph-call/tree/main/probing) involves training a Logistic Regression classifier to predict a property. The performance is used as a proxy to evaluate the model knowledge.
* [Neuron-level Analysis](https://github.com/fdalvi/NeuroX) [Durrani et al., 2020] allows retrieving a group of individual neurons that are most relevant to predict a linguistic property.
* [Contextual Correlation Analysis](https://github.com/johnmwu/contextual-corr-analysis/tree/master) [Wu et al., 2020] is a representation-level similarity measure that allows identifying pairs of layers of similar behavior. 

## Usage
We provide an [example](https://github.com/morphology-probing/morph-call/blob/main/examples/case-category-masks-probing.ipynb) of the experiment on **Masked Token** task (Case, German).

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
    --results_path=RESULTS_PATH
        Type: Optional[str]
        Default: None
        path to a folder to store the probing results and the model intermediate activations
    --model_architecture=MODEL_ARCHITECTURE
        Type: typ...
        Default: 'bert multilingual'
    --model_is_finetuned=MODEL_IS_FINETUNED
        Type: bool
        Default: False
        if to perform the experiment on the fine-tuned model
    --model_finetuned_path=MODEL_FINETUNED_PATH
        Type: Optional[str]
        Default: None
        (only if model_is_finetuned is True) path to store the fine-tuned model
    --model_finetuned_config_google_url=MODEL_FINETUNED_CONFIG_GOOGLE_URL
        Type: Optional[]
        Default: None
        (only if model_is_finetuned is True) the url of the fine-tuned model config if to be downloaded
    --model_finetuned_model_google_url=MODEL_FINETUNED_MODEL_GOOGLE_URL
        Type: Optional[]
        Default: None
        (only if model_is_finetuned is True) the url of the fine-tuned model weights if to be downloaded
    --model_is_random=MODEL_IS_RANDOM
        Type: bool
        Default: False
        if to perform the random initialization of the model
    --layers_to_probe=LAYERS_TO_PROBE
        Type: List
        Default: 'all'
        (either "all" or list w. possible numbers from 0 to 11) -- model layers to probe. e.g.: [1, 3, 11], or "all"
    --train_n_sentences=TRAIN_N_SENTENCES
        Type: int
        Default: 1500
        number of sentences used to train the probing classifier
    --test_n_sentences=TEST_N_SENTENCES
        Type: int
        Default: 1000
        number of sentences used to evaluate the probing classifier
    --dev_n_sentences=DEV_N_SENTENCES
        Type: int
        Default: 0
        DEPRECATED
```

## References
* Nadir Durrani, Hassan Sajjad, Fahim Dalvi, and Yonatan  Belinkov. 2020. Analyzing Individual Neurons  in  Pre-trained  Language  Models.
* John M Wu, Yonatan Belinkov, Hassan Sajjad, Nadir Durrani, Fahim Dalvi, and James Glass. 2020. Similarity analysis of contextual word representation models.


## Cite
The paper is accepted to the 3rd Workshop of the ACL Special Interest Group on Typology (SIGTYP) at NAACL, 2021. The title follows the morphing calls that were used by Power Rangers to call their powers.
