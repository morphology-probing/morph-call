import logging
import random
from collections import defaultdict
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

import constants

logger = logging.getLogger(f"{constants.LOGGER_NAME}.baseline")


class FeatureExtractor:
    def __init__(self):
        self.tokenizer = ToktokTokenizer()
        self.word_vectorizer = TfidfVectorizer(
            tokenizer=self.tokenizer.tokenize, lowercase=True, analyzer="word"
        )
        self.char_vectorizer = TfidfVectorizer(
            tokenizer=self.tokenizer.tokenize,
            lowercase=True,
            analyzer="char",
            ngram_range=(1, 3),
        )

    def create_tfidf_vectors(
        self,
        vectorizer: TfidfVectorizer,
        df: pd.DataFrame,
        mode: str,
        df_field: str = constants.WORD_COLNAME,
    ):
        vectors = (
            vectorizer.fit_transform(df[df_field])
            if mode == "train"
            else vectorizer.transform(df[df_field])
        )
        return vectors

    def create_char_num_vectors(
        self, df: pd.DataFrame, df_field: str = constants.WORD_COLNAME
    ):
        char_num_array = [len(w) for w in df[df_field]]
        return np.array(char_num_array).reshape(-1, 1)


class Baseline:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.set_seed()

    def set_seed(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def folded_fit_predict(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        df_baseline_features_src_field: str,
        tgt_colname: str,
        n_folds: int = constants.N_FOLDS,
    ) -> Dict[str, List[Dict[Any, float]]]:
        logger.debug("Creating features for train_df")
        train_features = {
            "char_number": self.feature_extractor.create_char_num_vectors(train_df),
            "tfidf_word": self.feature_extractor.create_tfidf_vectors(
                vectorizer=self.feature_extractor.word_vectorizer,
                df=train_df,
                mode="train",
                df_field=df_baseline_features_src_field,
            ),
            "tfidf_char_ngrams": self.feature_extractor.create_tfidf_vectors(
                vectorizer=self.feature_extractor.char_vectorizer,
                df=train_df,
                mode="train",
                df_field=df_baseline_features_src_field,
            ),
        }
        logger.debug("Creating features for test_df")
        test_features = {
            "char_number": self.feature_extractor.create_char_num_vectors(test_df),
            "tfidf_word": self.feature_extractor.create_tfidf_vectors(
                vectorizer=self.feature_extractor.word_vectorizer,
                df=test_df,
                mode="test",
                df_field=df_baseline_features_src_field,
            ),
            "tfidf_char_ngrams": self.feature_extractor.create_tfidf_vectors(
                vectorizer=self.feature_extractor.char_vectorizer,
                df=test_df,
                mode="test",
                df_field=df_baseline_features_src_field,
            ),
        }
        logger.debug("Predicting...")
        x_true = train_df[tgt_colname].to_numpy()
        y_true = test_df[tgt_colname].to_numpy()

        scores = defaultdict(list)

        for mode, feature in tqdm(
            train_features.items(),
            total=len(train_features),
            desc="calculating baseline probing scores",
        ):

            for fold_ix in range(n_folds):
                fold_ix += 1
                train_shuffled_index = list(range(len(x_true)))
                random.shuffle(train_shuffled_index)
                clf = LogisticRegression(n_jobs=1, verbose=0).fit(
                    feature[train_shuffled_index], x_true[train_shuffled_index]
                )
                pred = clf.predict(test_features[mode])

                scores[mode].append(
                    {"fold": fold_ix, "roc_auc": roc_auc_score(y_true, pred)}
                )

        return scores
