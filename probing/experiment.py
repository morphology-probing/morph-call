import datetime
import json
import logging
import os
import random
import re
import sqlite3
from collections import defaultdict
from typing import List, Tuple, Union, Any, Dict

import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

import constants
import defaults
from baseline import Baseline
from dataset_utils import create_dataset
from utils import load_model_and_tokenizer, calc_infer


class Experiment:
    def __init__(self):
        self._logger = logging.getLogger(
            f"{constants.LOGGER_NAME}.{self.__class__.__name__}"
        )
        self._baseline = None
        self.layer2probing_classifier = dict()

    @property
    def baseline(self):
        if self._baseline is None:
            self._baseline = Baseline()
        return self._baseline

    def _run(
        self,
        dataset_path: str,
        layers_to_probe="all",
        model_architecture: constants.MODELS_ARCHITECTURES_T = defaults.DEFAULT_MODEL_ARCHITECTURE,
        model_is_finetuned: bool = defaults.DEFAULT_WHETHER_TO_USE_FINETUNED_MODEL_OR_NOT,
        model_finetuned_path: str = None,
        model_is_random: bool = defaults.DEFAULT_WHETHER_TO_REINITIALIZE_MODEL_W_RANDOM_WEIGHTS_OR_NOT,
        results_dir_path: str = ".",
        train_size: int = None,
        test_size: int = None,
        val_size: int = None,
    ):
        """
        run inference and probing

        :param dataset_path: path to the probing dataset
        :param layers_to_probe: (either "all" or list w. possible numbers from 0 to 11) -- model
          layers to perform activations probing onto. example values: [1, 3, 11] or [0, 1, 2] or "all"
        :param model_architecture: one of ["multilingual_bert", "XLMR"] ("XLMR" support is coming soon)
        :param model_is_finetuned: whether to perform experiment on finetuned model or not
        :param model_finetuned_path: (only if model_is_finetuned is True)
          path to folder where the finetuned model files are stored. or
          url to such a folder in the web (e.g. http://files.deeppavlov.ai/morph-probing/models/bert/ru)
          defaults to "finetuned_model/"
        :param model_is_random: whether to reinitialize weights of model with random numbers or not
        :param results_dir_path: path to folder to store probing results and model inference activations
        :param train_size: the size to reduce train subsample to
        :param test_size: the size to reduce test subsample to
        :param val_size: the size to reduce val subsample to
        """
        self._logger.debug(
            f"BEFORE run() "
            f"layers_to_probe = {layers_to_probe} "
            f"dataset_path = {dataset_path} "
            f"model_architecture = {model_architecture} "
            f"model_is_finetuned = {model_is_finetuned} "
            f"model_finetuned_path = {model_finetuned_path} "
            f"model_is_random = {model_is_random} "
            f"results_dir_path = {results_dir_path} "
            f"train_size = {train_size} "
            f"test_size = {test_size} "
            f"val_size = {val_size} "
        )
        inference_results, is_word_level = self._infer(
            layers_to_probe=layers_to_probe,
            dataset_path=dataset_path,
            model_architecture=model_architecture,
            model_is_finetuned=model_is_finetuned,
            model_finetuned_path=model_finetuned_path,
            model_is_random=model_is_random,
            results_dir_path=results_dir_path,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
        )
        probing_scores = self._probe(
            inference_results[0][1], inference_results[1:], results_dir_path
        )

        results_dict = {"probing": probing_scores}
        if is_word_level:
            baseline_scores = self._baseline_probe(
                inference_results[0][1], inference_results[1:], results_dir_path
            )
            results_dict["baseline"] = baseline_scores

        results_json_path = os.path.join(results_dir_path, "results.json")
        with open(results_json_path, "w", encoding="utf-8") as results_f:
            self._logger.info(f"saving experiment results to {results_json_path}")
            json.dump(results_dict, results_f)

        self._logger.debug(
            f"AFTER run() "
            f"layers_to_probe = {layers_to_probe} "
            f"dataset_path = {dataset_path} "
            f"model_architecture = {model_architecture} "
            f"model_is_finetuned = {model_is_finetuned} "
            f"model_finetuned_path = {model_finetuned_path} "
            f"model_is_random = {model_is_random} "
            f"results_dir_path = {results_dir_path} "
            f"train_size = {train_size} "
            f"test_size = {test_size} "
            f"val_size = {val_size} "
            f"results_dict = {results_dict}"
        )

    def infer_and_probe(
        self,
        dataset_path: str,
        layers_to_probe="all",
        model_architecture: constants.MODELS_ARCHITECTURES_T = defaults.DEFAULT_MODEL_ARCHITECTURE,
        model_is_finetuned: bool = defaults.DEFAULT_WHETHER_TO_USE_FINETUNED_MODEL_OR_NOT,
        model_finetuned_path: str = None,
        model_is_random: bool = defaults.DEFAULT_WHETHER_TO_REINITIALIZE_MODEL_W_RANDOM_WEIGHTS_OR_NOT,
        results_dir_path: str = None,
        train_size: int = None,
        test_size: int = None,
        val_size: int = None,
    ):
        """
        run inference and probing

        :param dataset_path: path to the probing dataset
        :param layers_to_probe: (either "all" or list w. possible numbers from 0 to 11) -- model
          layers to perform activations probing onto. example values: [1, 3, 11] or [0, 1, 2] or "all"
        :param model_architecture: one of ["multilingual_bert"] ("XLMR" support is coming soon)
        :param model_is_finetuned: whether to perform experiment on finetuned model or not
        :param model_finetuned_path: (only if model_is_finetuned is True)
          path to folder where the finetuned model files are stored. or
          url to such a folder in the web (e.g. http://files.deeppavlov.ai/morph-probing/models/bert/ru)
          defaults to "finetuned_model/"
        :param model_is_random: whether to reinitialize weights of model with random numbers or not
        :param results_dir_path: path to folder to store probing results and model inference activations
        :param train_size: the size to reduce train subsample to
        :param test_size: the size to reduce test subsample to
        :param val_size: the size to reduce val subsample to
        """

        if results_dir_path is None:
            results_dir_path = self._configure_default_results_path(
                dataset_path=dataset_path,
                model_architecture=model_architecture,
                is_finetuned=model_is_finetuned,
                is_random=model_is_random,
            )
        os.makedirs(results_dir_path, exist_ok=True)
        logs_path = self.configure_logger(results_dir_path)
        logging.getLogger(constants.LOGGER_NAME).info(f"WRITING LOGS TO {logs_path}")

        self._run(
            dataset_path,
            layers_to_probe,
            model_architecture,
            model_is_finetuned,
            model_finetuned_path,
            model_is_random,
            results_dir_path,
            train_size,
            test_size,
            val_size,
        )

    def infer(
        self,
        dataset_path: str,
        layers_to_probe="all",
        model_architecture: constants.MODELS_ARCHITECTURES_T = defaults.DEFAULT_MODEL_ARCHITECTURE,
        model_is_finetuned: bool = defaults.DEFAULT_WHETHER_TO_USE_FINETUNED_MODEL_OR_NOT,
        model_finetuned_path: str = None,
        model_is_random: bool = defaults.DEFAULT_WHETHER_TO_REINITIALIZE_MODEL_W_RANDOM_WEIGHTS_OR_NOT,
        results_dir_path: str = None,
        train_size: int = None,
        test_size: int = None,
        val_size: int = None,
    ):
        """perform model inference on probing dataset

        :param dataset_path: path to the probing dataset
        :param layers_to_probe: (either "all" or list w. possible numbers from 0 to 11) -- model
          layers to perform activations probing onto. example values: [1, 3, 11] or [0, 1, 2] or "all"
        :param model_architecture: one of ["multilingual_bert"] ("XLMR" support is coming soon)
        :param model_is_finetuned: whether to perform experiment on finetuned model or not
        :param model_finetuned_path:  (only if model_is_finetuned is True)
          path to folder where the finetuned model files are stored. or
          url to such a folder in the web (e.g. http://files.deeppavlov.ai/morph-probing/models/bert/ru)
          defaults to "finetuned_model/"
        :param model_is_random: whether to reinitialize weights of model with random numbers or not
        :param results_dir_path: path to folder to store probing results and model inference activations
        :param train_size: the size to reduce train subsample to
        :param test_size: the size to reduce test subsample to
        :param val_size: the size to reduce val subsample to
        """

        if results_dir_path is None:
            results_dir_path = self._configure_default_results_path(
                dataset_path=dataset_path,
                model_architecture=model_architecture,
                is_finetuned=model_is_finetuned,
                is_random=model_is_random,
            )
        os.makedirs(results_dir_path, exist_ok=True)
        logs_path = self.configure_logger(results_dir_path)
        logging.getLogger(constants.LOGGER_NAME).info(f"WRITING LOGS TO {logs_path}")

        self._infer(
            dataset_path,
            layers_to_probe,
            model_architecture,
            model_is_finetuned,
            model_finetuned_path,
            model_is_random,
            results_dir_path,
            train_size,
            test_size,
            val_size,
        )

    def probe(
        self,
        train_probing_db_path: str,
        test_probing_datasets: Union[List[Tuple[str, str]], str],
        results_dir_path: str = None,
    ):
        """perform probing

        :param train_probing_db_path: path to db w. activations on train probing dataset
        :param test_probing_datasets: either path to db w. activations on test probing dataset
          or list of tuples <probing subsample name, path to db w. activations ot probing subsample dataset>
        :param results_dir_path: path to folder to store probing results and model inference activations
        :return: Nested dict of structure: probing_dataset_title -> layer_num -> probing_score
        """

        if results_dir_path is None:
            results_dir_path = self._configure_default_results_path(dataset_path="")
        os.makedirs(results_dir_path, exist_ok=True)
        logs_path = self.configure_logger(results_dir_path)
        logging.getLogger(constants.LOGGER_NAME).info(f"WRITING LOGS TO {logs_path}")

        self._probe(train_probing_db_path, test_probing_datasets, results_dir_path)

    def baseline(
        self,
        train_probing_db_path: str,
        test_probing_datasets: Union[List[Tuple[str, str]], str],
        results_dir_path: str = None,
    ):
        """calculate probing baseline scores

        :param train_probing_db_path: path to db w. train probing dataset
        :param test_probing_datasets: either path to db w. on test probing dataset
          or list of tuples <probing subsample name, path to db w. probing subsample dataset>
        :param results_dir_path: path to folder to store probing results and model inference activations
        :return: Nested dict of structure: probing_dataset_title -> baseline_metric -> probing_score
        """
        if results_dir_path is None:
            results_dir_path = self._configure_default_results_path(dataset_path="")
        os.makedirs(results_dir_path, exist_ok=True)
        logs_path = self.configure_logger(results_dir_path)
        logging.getLogger(constants.LOGGER_NAME).info(f"WRITING LOGS TO {logs_path}")

        self._baseline_probe(
            train_probing_db_path, test_probing_datasets, results_dir_path
        )

    def _infer(
        self,
        dataset_path: str,
        layers_to_probe="all",
        model_architecture: constants.MODELS_ARCHITECTURES_T = defaults.DEFAULT_MODEL_ARCHITECTURE,
        model_is_finetuned: bool = defaults.DEFAULT_WHETHER_TO_USE_FINETUNED_MODEL_OR_NOT,
        model_finetuned_path: str = None,
        model_is_random: bool = defaults.DEFAULT_WHETHER_TO_REINITIALIZE_MODEL_W_RANDOM_WEIGHTS_OR_NOT,
        results_dir_path: str = ".",
        train_size: int = None,
        test_size: int = None,
        val_size: int = None,
    ):
        """perform model inference on probing dataset

        :param dataset_path: path to the probing dataset
        :param layers_to_probe: (either "all" or list w. possible numbers from 0 to 11) -- model
          layers to perform activations probing onto. example values: [1, 3, 11] or [0, 1, 2] or "all"
        :param model_architecture: one of ["multilingual_bert"] ("XLMR" support is coming soon)
        :param model_is_finetuned: whether to perform experiment on finetuned model or not
        :param model_finetuned_path:  (only if model_is_finetuned is True)
          path to folder where the finetuned model files are stored. or
          url to such a folder in the web (e.g. http://files.deeppavlov.ai/morph-probing/models/bert/ru)
          defaults to "finetuned_model/"
        :param model_is_random: whether to reinitialize weights of model with random numbers or not
        :param results_dir_path: path to folder to store probing results and model inference activations
        :param train_size: the size to reduce train subsample to
        :param test_size: the size to reduce test subsample to
        :param val_size: the size to reduce val subsample to
        """
        self._logger.debug(
            f"BEFORE _infer() "
            f"layers_to_probe = {layers_to_probe} "
            f"dataset_path = {dataset_path} "
            f"model_architecture = {model_architecture} "
            f"model_is_finetuned = {model_is_finetuned} "
            f"model_finetuned_path = {model_finetuned_path} "
            f"model_is_random = {model_is_random} "
            f"results_dir_path = {results_dir_path} "
            f"train_size = {train_size} "
            f"test_size = {test_size} "
            f"val_size = {val_size} "
        )

        model, tokenizer = load_model_and_tokenizer(
            model_architecture,
            model_is_random,
            model_is_finetuned,
            model_finetuned_path,
        )

        os.makedirs(results_dir_path, exist_ok=True)

        layers_to_probe = self._set_up_layers_to_probe(
            layers_to_probe, model_architecture
        )

        dataset_titles, datasets, is_word_level = create_dataset(
            dataset_path,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            mask_token=tokenizer.mask_token if constants.DO_MASKING else None,
        )

        sentence_ix2features_paths = []
        for dataset_title, dataset in zip(dataset_titles, datasets):

            if constants.PROBING_COLUMN_NAME not in dataset.columns:
                self._logger.warning(
                    f"INSIDE _infer() column '{constants.PROBING_COLUMN_NAME}' was not in {dataset_title} dataset. "
                    "Added with always True value"
                )
                dataset[constants.PROBING_COLUMN_NAME] = True

            self._logger.debug(
                f"INSIDE _infer() processing {dataset_title} dataset of shape {dataset.shape}"
            )

            dataset_to_probe_name_without_extension = os.path.splitext(
                os.path.split(dataset_path)[1]
            )[0]
            dataset_to_probe_name = (
                dataset_to_probe_name_without_extension + dataset_title + "_db"
            )
            model_infers_dataset_db_path = os.path.join(
                results_dir_path, dataset_to_probe_name
            )

            sentence_ix2features_path = calc_infer(
                model=model,
                tokenizer=tokenizer,
                dataset_to_infer_to=dataset,
                layer_nums=layers_to_probe,
                number_of_sentences=None,
                is_word_level=is_word_level,
                path_for_db=model_infers_dataset_db_path,
            )

            self._logger.info(
                f"INSIDE _infer() activations for {dataset_title} stored in {sentence_ix2features_path}"
            )

            sentence_ix2features_paths.append(
                (dataset_title, sentence_ix2features_path)
            )

        self._logger.debug(
            f"AFTER _infer() "
            f"layers_to_probe = {layers_to_probe} "
            f"dataset_path = {dataset_path} "
            f"model_architecture = {model_architecture} "
            f"model_is_finetuned = {model_is_finetuned} "
            f"model_finetuned_path = {model_finetuned_path} "
            f"model_is_random = {model_is_random} "
            f"results_dir_path = {results_dir_path} "
            f"train_size = {train_size} "
            f"test_size = {test_size} "
            f"val_size = {val_size} "
            f"sentence_ix2features_paths = {sentence_ix2features_paths} "
            f"is_word_level = {is_word_level}"
        )

        return sentence_ix2features_paths, is_word_level

    def _set_up_layers_to_probe(
        self, layers_to_probe, model_architecture: constants.MODELS_ARCHITECTURES_T
    ):
        if layers_to_probe == "all":
            architecture_layers_num = constants.MODELS_ARCHITECTURES2LAYERS_NUM[
                model_architecture
            ]
            layers_to_probe = list(range(architecture_layers_num))
            self._logger.debug(
                f"INSIDE _set_up_layers_to_probe passed layers_to_probe='all', "
                f"set up to {str(layers_to_probe)}"
            )
        return layers_to_probe

    def _probe(
        self,
        train_probing_db_path: str,
        test_probing_datasets: Union[List[Tuple[str, str]], str],
        results_dir_path: str = ".",
    ) -> Dict[str, Dict[int, List[Dict[Any, float]]]]:
        """perform probing

        :param train_probing_db_path: path to db w. activations on train probing dataset
        :param test_probing_datasets: either path to db w. activations on test probing dataset
          or list of tuples <probing subsample name, path to db w. activations ot probing subsample dataset>
        :param results_dir_path: path to folder to store probing results and model inference activations
        :return: Nested dict of structure: probing_dataset_title -> layer_num -> probing_score
        """

        self._logger.debug(
            f"BEFORE _probe() " f"test_probing_datasets = {test_probing_datasets} "
        )

        if isinstance(test_probing_datasets, str):
            test_probing_datasets = [(test_probing_datasets, test_probing_datasets)]

        train_db_conn = sqlite3.connect(train_probing_db_path)

        layers_to_probe = self._extract_layers_to_probe_from_inference_dataset_conn(
            train_db_conn
        )
        hidden_size = self._extract_hs(train_db_conn)

        probing_results = defaultdict(dict)
        for layer_num in layers_to_probe:
            layer_activations_table = f"hidden_{layer_num}_activations"
            activation_colnames = [
                f"unit_{unit_ix}_activation" for unit_ix in range(hidden_size)
            ]

            experiment_columns = activation_colnames + [constants.TARGET_COLNAME]
            self._logger.debug("BEFORE READ TRAIN DATA")
            train_baseline_data = pd.read_sql_query(
                f"""SELECT {constants.PROBING_ITEM_IX_COLNAME}, {', '.join(experiment_columns)} 
                                                        FROM {layer_activations_table}
                                                     ;""",
                train_db_conn,
            )
            self._logger.debug("AFTER READ TRAIN DATA")

            for (
                test_baseline_orig_path,
                test_baseline_data_path,
            ) in test_probing_datasets:
                cur_layer_accuracy = self._probe_single_entry(
                    experiment_columns,
                    layer_activations_table,
                    layer_num,
                    test_baseline_data_path,
                    train_baseline_data,
                )

                probing_results[test_baseline_orig_path][layer_num] = cur_layer_accuracy

        train_db_conn.close()

        results_json_path = os.path.join(results_dir_path, "probing_results.json")
        with open(results_json_path, "w", encoding="utf-8") as results_f:
            self._logger.info(f"saving experiment results to {results_json_path}")
            json.dump(probing_results, results_f)

        self._logger.debug(
            f"AFTER _probe() "
            f"test_probing_datasets = {test_probing_datasets} "
            f"probing_results = {probing_results} stored in {results_json_path}"
        )

        return probing_results

    def _shuf_probe_layer(
        self, train_baseline_data, test_baseline_data, clf: LogisticRegression = None
    ) -> Tuple[LogisticRegression, float]:
        self._logger.debug(f"BEFORE _shuf_probe_layer()")

        X_train = train_baseline_data[
            [
                c
                for c in train_baseline_data.columns
                if c
                not in {constants.TARGET_COLNAME, constants.PROBING_ITEM_IX_COLNAME}
            ]
        ].to_numpy()
        y_train = train_baseline_data[constants.TARGET_COLNAME].to_numpy()

        X_test = test_baseline_data[
            [
                c
                for c in test_baseline_data.columns
                if c
                not in {constants.TARGET_COLNAME, constants.PROBING_ITEM_IX_COLNAME}
            ]
        ].to_numpy()
        y_test = test_baseline_data[constants.TARGET_COLNAME].to_numpy()

        train_ixes = list(range(len(X_train)))
        random.shuffle(train_ixes)
        if clf is not None:
            logreg_trained = clf
            pred = logreg_trained.predict(X_test)
            roc_auc_score = metrics.roc_auc_score(y_test, pred)
        else:
            regs = [0.25, 0.5, 1, 2, 4]
            best = {"score": -1, "reg": regs[0], "model": None}

            classes = list(set(y_train))
            class_weight = compute_class_weight("balanced", classes, y_train)
            class_weight = dict(zip(classes, class_weight))

            for reg in regs:
                logreg_trained: LogisticRegression = LogisticRegression(
                    max_iter=constants.LOGERG_ITER, C=reg, class_weight=class_weight
                ).fit(X_train[train_ixes], y_train[train_ixes])
                pred = logreg_trained.predict(X_test)
                roc_auc_score = metrics.roc_auc_score(y_test, pred)
                if roc_auc_score > best["score"]:
                    best["score"] = roc_auc_score
                    best["model"] = logreg_trained
                    best["reg"] = reg
            logreg_trained = best["model"]
            roc_auc_score = best["score"]

        regularization_param = logreg_trained.get_params().get("C", None)
        class_weight = logreg_trained.get_params().get("class_weight", None)

        self._logger.debug(
            f"AFTER _shuf_probe_layer() "
            f"roc_auc_score = {roc_auc_score}, "
            f"regularization_param={regularization_param}, "
            f"class_weight={class_weight}"
        )
        return logreg_trained, roc_auc_score

    def _extract_layers_to_probe_from_inference_dataset_conn(
        self, inference_dataset_conn
    ):
        layer_num_pattern = r"hidden_(?P<layer_num>\d+)_activations"
        layers_to_probe = []

        cursor = inference_dataset_conn.cursor()
        # noinspection SqlResolve
        colnames = [
            desc[0]
            for desc in cursor.execute(
                f"SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        ]

        for train_colname in colnames:
            column_hidden_info_match = re.match(layer_num_pattern, train_colname)
            if column_hidden_info_match:
                column_layer_num = int(column_hidden_info_match["layer_num"])
                layers_to_probe.append(column_layer_num)
        return layers_to_probe

    def _baseline_probe(
        self,
        train_probing_db_path: str,
        test_probing_datasets: Union[List[Tuple[str, str]], str],
        results_dir_path: str = ".",
    ) -> Dict[str, Dict[str, Any]]:
        """calculate probing baseline scores

        :param train_probing_db_path: path to db w. train probing dataset
        :param test_probing_datasets: either path to db w. on test probing dataset
          or list of tuples <probing subsample name, path to db w. probing subsample dataset>
        :param results_dir_path: path to folder to store probing results and model inference activations
        :return: Nested dict of structure: probing_dataset_title -> baseline_metric -> probing_score
        """
        # inference_results_datasets: List[Tuple[str, str]]):
        self._logger.debug(
            f"BEFORE _baseline_probe() "
            f"train_probing_db_path = {train_probing_db_path} "
            f"test_probing_datasets = {test_probing_datasets}"
        )

        if isinstance(test_probing_datasets, str):
            test_probing_datasets = [(test_probing_datasets, test_probing_datasets)]

        baseline_feature_colname = constants.WORD_COLNAME
        baseline_target_colname = constants.TARGET_COLNAME
        train_baseline_data_conn = sqlite3.connect(train_probing_db_path)
        # noinspection SqlResolve
        train_baseline_data = pd.read_sql_query(
            f"SELECT {baseline_feature_colname}, {baseline_target_colname} "
            f"FROM {constants.PROBING_TABLE_NAME} "
            f"WHERE ok_for_baseline = 1",
            train_baseline_data_conn,
        )
        train_baseline_data_conn.close()
        probing_results = dict()
        for test_baseline_orig_path, test_baseline_data_path in test_probing_datasets:
            probing_results[
                test_baseline_orig_path
            ] = self._baseline_probe_single_dataset(
                baseline_feature_colname,
                baseline_target_colname,
                test_baseline_data_path,
                train_baseline_data,
            )

        results_json_path = os.path.join(
            results_dir_path, "baseline_probing_results.json"
        )
        with open(results_json_path, "w", encoding="utf-8") as results_f:
            self._logger.info(f"saving experiment results to {results_json_path}")
            json.dump(probing_results, results_f)

        self._logger.debug(
            f"AFTER _baseline_probe() "
            f"train_probing_db_path = {train_probing_db_path} "
            f"test_probing_datasets = {test_probing_datasets}"
            f"probing_results = {probing_results} stored in {results_json_path}"
        )
        return probing_results

    def _baseline_probe_single_dataset(
        self,
        baseline_feature_colname,
        baseline_target_colname,
        test_baseline_data_path,
        train_baseline_data,
    ):
        test_baseline_data_conn = sqlite3.connect(test_baseline_data_path)
        # noinspection SqlResolve
        test_baseline_data = pd.read_sql_query(
            f"SELECT {baseline_feature_colname}, {baseline_target_colname} "
            f"FROM {constants.PROBING_TABLE_NAME} "
            f"WHERE ok_for_baseline = 1",
            test_baseline_data_conn,
        )
        test_baseline_data_conn.close()
        probing_scores = self.baseline.folded_fit_predict(
            train_df=train_baseline_data,
            test_df=test_baseline_data,
            df_baseline_features_src_field=baseline_feature_colname,
            tgt_colname=baseline_target_colname,
        )
        return probing_scores

    def _probe_single_entry(
        self,
        experiment_columns,
        layer_activations_table,
        layer_num,
        test_baseline_data_path,
        train_baseline_data,
        n_folds: int = constants.N_FOLDS,
    ) -> List[Dict[Any, float]]:
        test_baseline_data_conn = sqlite3.connect(test_baseline_data_path)
        cur_layer_accuracy = list()

        self._logger.debug(
            f"INSIDE _probe() "
            f"started probing layer {layer_num} "
            f"in dataset {test_baseline_data_path} "
        )
        for fold_ix in range(n_folds):
            fold_ix += 1
            test_baseline_data = pd.read_sql_query(
                f"SELECT {constants.PROBING_ITEM_IX_COLNAME}, "
                f"       {', '.join(experiment_columns)} "
                f"FROM {layer_activations_table};",
                test_baseline_data_conn,
            )

            metric_name = "roc_auc"
            clf = self.layer2probing_classifier.get(layer_num)
            clf, metric_value = self._shuf_probe_layer(
                train_baseline_data, test_baseline_data, clf
            )
            self.layer2probing_classifier[layer_num] = clf
            cur_layer_accuracy.append({"fold": fold_ix, metric_name: metric_value})
            self._logger.debug(
                f"INSIDE _probe() (score {cur_layer_accuracy[-1][metric_name]}) "
                f"fold {fold_ix} of {n_folds}"
                f"finished probing layer {layer_num} "
                f"in dataset {test_baseline_data_path} "
            )

        self._logger.info(
            f"INSIDE _probe() (scores {cur_layer_accuracy}) "
            f"finished probing layer {layer_num} "
            f"in dataset {test_baseline_data_path} "
        )
        test_baseline_data_conn.close()
        return cur_layer_accuracy

    @staticmethod
    def _configure_default_results_path(
        dataset_path: str,
        model_architecture: str = None,
        is_finetuned: bool = None,
        is_random: bool = None,
    ):
        """default path to store results is ``"inference_results/year-month-day-hours-minutes-seconds"``"""

        results_path_time = datetime.datetime.now().strftime(
            "%F-%H-%M-%S"
        )  # year-month-day-hours-minutes-seconds
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        model_code = f"{dataset_name}_{results_path_time}"
        if model_architecture is not None:
            model_codes_shortening = {
                constants.BERT_MULTILINGUAL_A_NAME: "bert_multling",
                constants.XLMR_A_NAME: "xlmr",
            }
            if model_architecture in model_codes_shortening:
                model_code_shortening = model_codes_shortening[model_architecture]
                model_code += f"_{model_code_shortening}"
        if is_finetuned is not None:
            model_code += f"_finetuned{is_finetuned}"
        if is_random is not None:
            model_code += f"_random{is_random}"
        results_path_dir = "inference_results"
        results_path = os.path.join(results_path_dir, model_code)
        return results_path

    @staticmethod
    def configure_logger(results_path):
        logs_path = os.path.join(results_path, "experiment.log")
        handler = logging.FileHandler(logs_path)
        handler.setFormatter(logging.Formatter(constants.LOG_FORMAT))
        logging.getLogger(constants.LOGGER_NAME).addHandler(handler)
        return logs_path

    def _extract_hs(self, train_baseline_data_conn):
        cursor = train_baseline_data_conn.cursor()
        # noinspection SqlResolve
        colnames = [
            desc[0]
            for desc in cursor.execute(
                f"SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        ]

        col_to_use = None
        for train_colname in colnames:
            if train_colname.startswith("hidden"):
                col_to_use = train_colname

        unit_colnames = [
            desc[0]
            for desc in train_baseline_data_conn.execute(
                f"SELECT * FROM {col_to_use} LIMIT 1;"
            ).description
        ]
        unit_num_pattern = r"unit_(?P<unit_num>\d+)_activation"
        unit_nums = []
        for unit_colname in unit_colnames:
            column_hidden_info_match = re.match(unit_num_pattern, unit_colname)
            if column_hidden_info_match:
                unit_num = int(column_hidden_info_match["unit_num"])
                unit_nums.append(unit_num)

        return len(unit_nums)
