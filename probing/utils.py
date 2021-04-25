import gc
import logging
import os
import pickle
import random
import sqlite3
from collections import defaultdict
from typing import List, Dict, Any, Union

import numpy as np
import pandas as pd
import torch
import transformers
from tqdm.auto import tqdm

import constants

# import constants
import defaults
from morpho_dataset import MorphoDataset

tqdm.pandas()

logger = logging.getLogger(f"{constants.LOGGER_NAME}.utils")


def load_model_and_tokenizer(
    model_architecture: constants.MODELS_ARCHITECTURES_T,
    model_is_random: bool,
    model_is_finetuned: bool,
    model_finetuned_path: str,
):
    logger.debug(
        f"BEFORE load_model_and_tokenizer() "
        f"model_architecture = {model_architecture} "
        f"model_is_random = {model_is_random} "
        f"model_is_finetuned = {model_is_finetuned} "
        f"model_finetuned_path = {model_finetuned_path} "
    )

    assert model_architecture in constants.MODELS_ARCHITECTURES

    if model_is_finetuned:
        if model_finetuned_path is None:
            model_finetuned_path = defaults.DEFAULT_FINETUNED_MODEL_PATH

    if model_architecture == constants.BERT_MULTILINGUAL_A_NAME:
        TOKENIZER_C = transformers.BertTokenizer
        MODEL_C = transformers.BertModel
        CONFIG_C = transformers.BertConfig

        PRETRAINED_MODEL_NAME = "bert-base-multilingual-cased"
        tokenizer = TOKENIZER_C.from_pretrained(PRETRAINED_MODEL_NAME)
        if model_is_finetuned:
            config = CONFIG_C.from_pretrained(
                f"{model_finetuned_path}/config.json", output_hidden_states=True
            )
            model = MODEL_C.from_pretrained(
                f"{model_finetuned_path}/pytorch_model.bin", config=config
            ).cuda()
        else:
            config = CONFIG_C.from_pretrained(
                PRETRAINED_MODEL_NAME, output_hidden_states=True
            )
            model = MODEL_C.from_pretrained(PRETRAINED_MODEL_NAME, config=config).cuda()

    elif model_architecture == constants.XLMR_A_NAME:
        TOKENIZER_C = transformers.XLMRobertaTokenizer
        MODEL_C = transformers.XLMRobertaModel
        CONFIG_C = transformers.XLMRobertaConfig

        PRETRAINED_MODEL_NAME = "xlm-roberta-base"
        tokenizer = TOKENIZER_C.from_pretrained(PRETRAINED_MODEL_NAME)
        if model_is_finetuned:
            config = CONFIG_C.from_pretrained(
                f"{model_finetuned_path}/config.json", output_hidden_states=True
            )
            model = MODEL_C.from_pretrained(
                f"{model_finetuned_path}/pytorch_model.bin", config=config
            ).cuda()
        else:
            config = CONFIG_C.from_pretrained(
                PRETRAINED_MODEL_NAME, output_hidden_states=True
            )
            model = MODEL_C.from_pretrained(PRETRAINED_MODEL_NAME, config=config).cuda()

    elif model_architecture == constants.BERT_DISTIL_A_NAME:
        TOKENIZER_C = transformers.DistilBertTokenizer
        MODEL_C = transformers.DistilBertModel
        CONFIG_C = transformers.DistilBertConfig

        PRETRAINED_MODEL_NAME = "distilbert-base-multilingual-cased"
        tokenizer = TOKENIZER_C.from_pretrained(PRETRAINED_MODEL_NAME)
        if model_is_finetuned:
            config = CONFIG_C.from_pretrained(
                f"{model_finetuned_path}/config.json", output_hidden_states=True
            )
            model = MODEL_C.from_pretrained(
                f"{model_finetuned_path}/pytorch_model.bin", config=config
            ).cuda()
        else:
            config = CONFIG_C.from_pretrained(
                PRETRAINED_MODEL_NAME, output_hidden_states=True
            )
            model = MODEL_C.from_pretrained(PRETRAINED_MODEL_NAME, config=config).cuda()

    elif model_architecture == constants.MINILM_A_NAME:
        TOKENIZER_C = transformers.XLMRobertaTokenizer
        MODEL_C = transformers.BertModel
        CONFIG_C = transformers.BertConfig

        PRETRAINED_MODEL_NAME = "microsoft/Multilingual-MiniLM-L12-H384"
        tokenizer = TOKENIZER_C.from_pretrained(PRETRAINED_MODEL_NAME)
        if model_is_finetuned:
            config = CONFIG_C.from_pretrained(
                f"{model_finetuned_path}/config.json", output_hidden_states=True
            )
            model = MODEL_C.from_pretrained(
                f"{model_finetuned_path}/pytorch_model.bin", config=config
            ).cuda()
        else:
            config = CONFIG_C.from_pretrained(
                PRETRAINED_MODEL_NAME, output_hidden_states=True
            )
            model = MODEL_C.from_pretrained(PRETRAINED_MODEL_NAME, config=config).cuda()

    else:
        raise ValueError(
            f"model_architecture should be one of "
            + ", ".join(
                [
                    constants.BERT_MULTILINGUAL_A_NAME,
                    constants.BERT_DISTIL_A_NAME,
                    constants.XLMR_A_NAME,
                    constants.XLMR_A_NAME,
                ]
            )
        )

    if model_is_random:
        model.init_weights()

    logger.debug(
        f"AFTER load_model_and_tokenizer() "
        f"model_architecture = {model_architecture} "
        f"model_is_random = {model_is_random} "
        f"model_is_finetuned = {model_is_finetuned} "
        f"model_finetuned_path = {model_finetuned_path} "
    )

    return model, tokenizer


def orig_toks_ixes2bert_toks_ixes_mapping(
    sent_orig_tokenized, sent_token_ixes2remember, sent_bert_tokenized
) -> Dict[int, List[int]]:

    bert_tokenization_t = "bert_tokenization"
    xlmr_tokenization_t = "xlmr_tokenization"
    sentencepiece_prefix = "▁"
    if sent_bert_tokenized[0] == "[CLS]":
        tokenization_type = bert_tokenization_t
    elif sent_bert_tokenized[0] == "<s>":
        tokenization_type = xlmr_tokenization_t
    else:
        raise ValueError(
            f"could not understand the tokenizer for {sent_bert_tokenized}"
        )

    if tokenization_type == bert_tokenization_t:
        assert all(
            not orig_token.startswith("##") for orig_token in sent_orig_tokenized
        )
    elif tokenization_type == xlmr_tokenization_t:
        # assert any(orig_token.startswith(sentencepiece_prefix) for orig_token in sent_orig_tokenized)
        pass

    if tokenization_type == bert_tokenization_t:
        piece_starting_tokens_ixes = [
            tok_ix
            for tok_ix, tok in enumerate(sent_bert_tokenized)
            if (tok not in {"[CLS]", "[SEP]"} and not tok.startswith("##"))
            or tok == "[MASK]"
        ]
    elif tokenization_type == xlmr_tokenization_t:
        piece_starting_tokens_ixes = [
            tok_ix
            for tok_ix, tok in enumerate(sent_bert_tokenized)
            if tok.startswith(sentencepiece_prefix) or tok == "<mask>"
        ]

    assert len(piece_starting_tokens_ixes) == len(sent_orig_tokenized)

    if sent_token_ixes2remember == [constants.FAKE_TOK_IX]:
        res = {
            constants.FAKE_TOK_IX: list(
                range(piece_starting_tokens_ixes[0], len(sent_bert_tokenized) - 1)
            )
        }
        return res

    orig_toks2bert_toks_ixes_mapping = dict()

    orig_tok_ix = 0
    for orig_tok, respective_bert_tok_ix, respective_bert_toks_span_end in zip(
        sent_orig_tokenized,
        piece_starting_tokens_ixes,
        piece_starting_tokens_ixes[1:] + [len(sent_bert_tokenized) - 1],
    ):

        if orig_tok_ix in sent_token_ixes2remember:
            orig_toks2bert_toks_ixes_mapping[orig_tok_ix] = list(
                range(respective_bert_tok_ix, respective_bert_toks_span_end)
            )
        orig_tok_ix += 1

    return orig_toks2bert_toks_ixes_mapping


def calc_bert_token_ixes_to_remember(row):
    qvs = []
    try:
        res = []
        sent_orig_tokenized = row[constants.TOKENIZED_COLNAME]
        sent_token_ixes2remember = row[constants.TOKEN_IX_COLNAME]
        sent_bert_tokenized = row.bert_tokenized
        sentence_ix = row[constants.SENTENCE_IX_COLNAME]

        for word_ix, bert_tok_ixes in orig_toks_ixes2bert_toks_ixes_mapping(
            sent_orig_tokenized, sent_token_ixes2remember, sent_bert_tokenized
        ).items():
            # print(word_ix, bert_tok_ixes)
            for bert_tok_ix in bert_tok_ixes:
                res.append(bert_tok_ix)
                qvs.append((sentence_ix, word_ix, bert_tok_ix))

    except Exception as e:
        res = []
        # raise e
    return res, qvs


def prepare_tokenized_sentences_dataset(
    dataset_to_infer_to,
    tokenizer,
    sentence_text_colname=constants.SENTENCE_TEXT_COLNAME,
    probing_item_colname=constants.WORD_COLNAME,
    item_level_tokenization_colname=constants.TOKENIZED_COLNAME,
):
    # make DF with sentences, tokenized sentences and bert-tokenized-sentences
    # now sentences tokenization is available as a column
    # print(dataset_to_infer_to.columns, probing_item_colname)
    # print(dataset_to_infer_to[dataset_to_infer_to[constants.SENTENCE_IX_COLNAME]])
    logger.debug(
        f"BEFORE prepare_tokenized_sentences_dataset() "
        f"dataset_to_infer_to_shape = {dataset_to_infer_to.shape} "
        f"tokenizer = {tokenizer} "
        f"sentence_text_colname = {sentence_text_colname} "
        f"probing_item_colname = {probing_item_colname} "
        f"item_level_tokenization_colname = {item_level_tokenization_colname} "
    )
    sentences_tokenized = dataset_to_infer_to.groupby(
        by=constants.SENTENCE_IX_COLNAME, as_index=False
    ).agg(
        {
            probing_item_colname: list,
            constants.PROBING_COLUMN_NAME: list,
            constants.TOKEN_IX_COLNAME: list,
            constants.TOKENIZED_COLNAME: "first",
            sentence_text_colname: "first",
        }
    )

    sentences_tokenized: pd.DataFrame = sentences_tokenized[
        sentences_tokenized[constants.PROBING_COLUMN_NAME].apply(any)
    ]

    def foo(s):
        res = tokenizer.encode(s, is_pretokenized=True)
        if len(res) > 512:
            logger.info(f"too long, will be skipped: {s}")
        return res

    sentence_bert_tokenized_ixes = sentences_tokenized[
        constants.TOKENIZED_COLNAME
    ].progress_apply(foo)
    sentences_tokenized["bert_tokenized_ixes"] = sentence_bert_tokenized_ixes
    sentences_tokenized = sentences_tokenized[
        sentences_tokenized["bert_tokenized_ixes"].apply(len) <= 512
    ]
    logger.debug(
        f"INSIDE prepare_tokenized_sentences_dataset() started convert_ids_to_tokens()"
    )
    if logger.isEnabledFor(logging.DEBUG):
        sentence_bert_tokenized = (
            sentences_tokenized.bert_tokenized_ixes.progress_apply(
                tokenizer.convert_ids_to_tokens
            )
        )
    else:
        sentence_bert_tokenized = sentences_tokenized.bert_tokenized_ixes.apply(
            tokenizer.convert_ids_to_tokens
        )
    logger.debug(
        f"INSIDE prepare_tokenized_sentences_dataset() finished convert_ids_to_tokens()"
    )
    sentences_tokenized["bert_tokenized"] = sentence_bert_tokenized

    sentences_tokenized["bert_token_ixes_to_remember"] = None
    dfs_large = []
    for row_ix, row in sentences_tokenized.iterrows():
        bert_token_ixes_to_remember, dfs = calc_bert_token_ixes_to_remember(row)
        sentences_tokenized.at[
            row_ix, "bert_token_ixes_to_remember"
        ] = bert_token_ixes_to_remember
        dfs_large.extend(dfs)

    sentences_tokenized = sentences_tokenized[
        sentences_tokenized.bert_token_ixes_to_remember.apply(any)
    ]
    sentences_tokenized.reset_index(inplace=True)

    logger.debug(
        f"AFTER prepare_tokenized_sentences_dataset() "
        f"dataset_to_infer_to_shape = {dataset_to_infer_to.shape} "
        f"tokenizer = {tokenizer} "
        f"sentence_text_colname = {sentence_text_colname} "
        f"probing_item_colname = {probing_item_colname} "
        f"item_level_tokenization_colname = {item_level_tokenization_colname} "
        f"sentences_tokenized.shape = {sentences_tokenized.shape} "
    )
    return sentences_tokenized, dfs_large


def infer_on_sentences(
    dataloader, model, layer_nums: List[int], inference_db_conn
) -> Dict[str, Any]:
    model.eval()

    sentixtokix2wordix = (
        pd.read_sql(
            f"SELECT "
            f"{constants.SENTENCE_IX_COLNAME}, "
            f"{constants.TOKEN_IX_COLNAME}, "
            f"{constants.BERT_IX_COLNAME} "
            f"FROM {constants.O2B_TABLE_NAME};",
            inference_db_conn,
        )
        .groupby(by=[constants.SENTENCE_IX_COLNAME, constants.BERT_IX_COLNAME])
        .agg("first")[constants.TOKEN_IX_COLNAME]
        .to_dict()
    )
    sentixwordpos2wordid = (
        pd.read_sql(
            f"SELECT "
            f"{constants.SENTENCE_IX_COLNAME}, "
            f"{constants.TOKEN_IX_COLNAME}, "
            f"{constants.PROBING_ITEM_IX_COLNAME} "
            f"FROM {constants.PROBING_TABLE_NAME};",
            inference_db_conn,
        )
        .groupby(by=[constants.SENTENCE_IX_COLNAME, constants.TOKEN_IX_COLNAME])
        .agg("first")[constants.PROBING_ITEM_IX_COLNAME]
        .to_dict()
    )
    wordix2target = pd.read_sql(
        f"SELECT {constants.PROBING_ITEM_IX_COLNAME}, {constants.TARGET_COLNAME} "
        f"FROM {constants.PROBING_TABLE_NAME};",
        inference_db_conn,
        index_col=constants.PROBING_ITEM_IX_COLNAME,
    )[constants.TARGET_COLNAME].to_dict()

    inference_db_cursor = inference_db_conn.cursor()

    hl_shape: int = get_hl_shape(model)

    for layer_num in layer_nums:
        inference_db_cursor.execute(
            f"CREATE TABLE hidden_{layer_num}_activations ("
            f"  {constants.PROBING_ITEM_IX_COLNAME} integer PRIMARY KEY , "
            f"  {constants.TARGET_COLNAME} integer NOT NULL , "
            f"  {', '.join(f'unit_{unit_ix}_activation real ' for unit_ix in range(hl_shape))} "
            f");"
        )

    with torch.no_grad():
        db_update_queries = {
            layer_num: (
                f"INSERT INTO hidden_{layer_num}_activations ("
                f"  {constants.PROBING_ITEM_IX_COLNAME}, "
                f"  {constants.TARGET_COLNAME}, "
                f"  {', '.join(f'unit_{unit_ix}_activation' for unit_ix in range(hl_shape))}"
                f") VALUES ({', '.join('?' * (hl_shape + 2))});"
            )
            for layer_num in layer_nums
        }
        db_updates = {layer_num: defaultdict(list) for layer_num in layer_nums}

        for sentences_ids, toks_ids_to_remember, token_ids, attention_ids in tqdm(
            dataloader, desc="Infer on sentences"
        ):

            hidden_states = model(token_ids, attention_ids, return_dict=True)[
                "hidden_states"
            ]

            sentence_id2layer2features = defaultdict(dict)
            for layer_num in layer_nums:

                hl_ix = 1 + layer_num
                hl_acts_batch = hidden_states[hl_ix].cpu()

                attention_ids = attention_ids.cpu()

                hl_acts_batch_Zpadded = hl_acts_batch.permute([2, 0, 1]) * attention_ids
                hl_acts_batch_Zpadded = hl_acts_batch_Zpadded.permute([1, 2, 0])

                for batch_entry_ix, batch_entry_acts in enumerate(
                    hl_acts_batch_Zpadded
                ):
                    sentence_id = int(sentences_ids[batch_entry_ix])

                    curr_hl_activations_on_tokens = dict()
                    for token_ix, batch_entry_token_acts in enumerate(batch_entry_acts):
                        if not batch_entry_token_acts.nonzero().nelement():
                            break
                        curr_token_features = batch_entry_token_acts.cpu()
                        word_ix = sentixtokix2wordix.get((sentence_id, token_ix))
                        probing_item_ix = sentixwordpos2wordid.get(
                            (sentence_id, word_ix)
                        )
                        if not word_ix:
                            continue

                        db_updates[layer_num][probing_item_ix].append(
                            curr_token_features.numpy().tolist()
                        )

            for l_num in layer_nums:
                updates = []
                for word_ix in db_updates[l_num].keys():
                    try:
                        updates.append(
                            [word_ix, wordix2target[word_ix]]
                            + np.mean(db_updates[l_num][word_ix], axis=0).tolist()
                        )
                    except Exception as e:
                        raise e
                inference_db_cursor.executemany(db_update_queries[l_num], updates)
                db_updates[l_num] = defaultdict(list)

            inference_db_conn.commit()
            del attention_ids
            gc.collect()


def get_hl_shape(
    model: Union[
        transformers.BertModel,
        transformers.DistilBertModel,
        transformers.XLMRobertaModel,
    ]
):
    return int(model.config.hidden_size)


def bert_tokenized_features2orig_tokenization_features(
    sentence_ix2features,
    sentences_tokenization_df: pd.DataFrame,
    probing_dataset,
    path_for_db,
) -> str:
    first_sentence_features = next(sentence_ix2features.values())
    layer_nums = list(first_sentence_features.keys())

    database_path = os.path.join(
        path_for_db, f"{constants.AGGREGATED_DB_NAME}.sqlite_db"
    )
    activation_logs_database_conn = sqlite3.connect(database_path)

    sentence_text2sentence_ix = {
        sentence_text: sentence_ix
        for sentence_ix, sentence_text in sentences_tokenization_df[
            constants.SENTENCE_TEXT_COLNAME
        ].iteritems()
    }

    resulting_dataset = probing_dataset.copy()
    resulting_dataset.drop(columns=constants.TOKENIZED_COLNAME, inplace=True)

    # the column will be used to filter out rows that were failed to be processed for some reason
    resulting_dataset["issues"] = None
    for layer_num in layer_nums:
        # here we'll store the activations of model on the given data
        resulting_dataset[
            f"hidden_{str(layer_num)}_activations"
        ] = None  # [[] for _ in range(len(resulting_dataset))]

    resulting_dataset.to_sql(
        name=constants.AGGREGATED_DB_NAME,
        con=activation_logs_database_conn,
        index_label=constants.INDEX_COLNAME,
    )

    activation_logs_database_cursor = activation_logs_database_conn.cursor()

    curr_sentence_text = None
    # sentence_starting_word_ix_in_dataset = None
    for probing_entry_ix, item_row in tqdm(
        probing_dataset.iterrows(),
        total=probing_dataset.shape[0],
        desc="aggregate computed activations properly",
    ):
        sentence_text = item_row[constants.SENTENCE_TEXT_COLNAME]
        word_ix_in_sentence = item_row[constants.TOKEN_IX_COLNAME]

        activation_logs_database_conn.commit()

        sentence_id = sentence_text2sentence_ix.get(sentence_text, "")

        sentence_activations = sentence_ix2features.get(str(sentence_id))
        queries_to_execute = []

        if not sentence_activations:
            activation_logs_database_cursor.execute(
                f"UPDATE {constants.AGGREGATED_DB_NAME} "
                f"SET issues = 'not in sample' "
                f"WHERE {constants.INDEX_COLNAME} = {probing_entry_ix};"
            )
            continue
        try:
            sentence_tokenization_info = sentences_tokenization_df.loc[sentence_id]
            curr_word_bert_tokenization_ixes = bert_tokens_ixes_for_word(
                sentence_tokenization_info, word_ix_in_sentence
            )
        except:
            activation_logs_database_cursor.execute(
                f"UPDATE {constants.AGGREGATED_DB_NAME} "
                f"SET issues = 'tokenization issues' "
                f"WHERE {constants.INDEX_COLNAME} = {probing_entry_ix};"
            )
            continue

        tokens_elems_activations_on_layers = defaultdict(list)
        for layer_num in layer_nums:
            hidden_layer_activations_colname = f"hidden_{str(layer_num)}_activations"
            for token_ix in curr_word_bert_tokenization_ixes:
                token_activations = sentence_activations[layer_num][token_ix]
                tokens_elems_activations_on_layers[layer_num].append(token_activations)

        for layer_num in layer_nums:
            token_elems_activations_pkl = pickle.dumps(
                tokens_elems_activations_on_layers[layer_num]
            )
            hidden_layer_activations_colname = f"hidden_{str(layer_num)}_activations"
            activation_logs_database_cursor.execute(
                f"UPDATE {constants.AGGREGATED_DB_NAME} "
                f"SET {hidden_layer_activations_colname} = ? "
                f"WHERE {constants.INDEX_COLNAME} = {probing_entry_ix};",
                (token_elems_activations_pkl,),
            )

    activation_logs_database_conn.commit()
    activation_logs_database_conn.close()
    sentence_ix2features.close()

    return database_path


def bert_tokens_ixes_for_word(sentence_tokenization_info, word_ix_in_sentence):
    sent_orig_tokenized = sentence_tokenization_info[constants.TOKENIZED_COLNAME]
    sent_token_ixes2remember = sentence_tokenization_info[constants.TOKEN_IX_COLNAME]
    sent_bert_tokenized = sentence_tokenization_info.bert_tokenized
    orig_tokenization_ixes2bert_tokenization_ixes = (
        orig_toks_ixes2bert_toks_ixes_mapping(
            sent_orig_tokenized, sent_token_ixes2remember, sent_bert_tokenized
        )
    )
    curr_word_bert_tokenization_ixes = orig_tokenization_ixes2bert_tokenization_ixes[
        word_ix_in_sentence
    ]
    return curr_word_bert_tokenization_ixes


def calc_infer(
    model,
    tokenizer,
    dataset_to_infer_to: pd.DataFrame,
    layer_nums: List[int],
    number_of_sentences: int = None,
    is_word_level: bool = True,
    path_for_db: str = ".",
) -> str:
    logger.debug(
        f"BEFORE calc_infer() "
        f"model = {model} "
        f"tokenizer = {tokenizer} "
        f"dataset_to_infer_to = {dataset_to_infer_to} "
        f"layer_nums = {layer_nums} "
        f"is_word_level = {is_word_level} "
        f"number_of_sentences = {number_of_sentences} "
        f"path_for_db = {path_for_db}"
    )

    sentences_tokenization_df, dfs_large = prepare_tokenized_sentences_dataset(
        dataset_to_infer_to,
        tokenizer,
        probing_item_colname=(
            constants.WORD_COLNAME
            if is_word_level
            else constants.DUPLICATE_SENTENCE_TEXT_COLNAME
        ),
    )
    dataloader = configure_dataloader(
        number_of_sentences, sentences_tokenization_df, tokenizer
    )

    db_dir = os.path.join(path_for_db, "db/")
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "db.sqlite")

    inference_db_conn = sqlite3.connect(db_path)

    dataset_to_infer_to[
        [c for c in dataset_to_infer_to.columns if c != constants.TOKENIZED_COLNAME]
    ].to_sql(constants.PROBING_TABLE_NAME, inference_db_conn)

    cursor = inference_db_conn.cursor()

    q = (
        f"CREATE TABLE {constants.O2B_TABLE_NAME} ("
        f"id integer PRIMARY KEY , "
        f"{constants.SENTENCE_IX_COLNAME} integer NOT NULL, "
        f"{constants.TOKEN_IX_COLNAME} integer NOT NULL, "
        f"{constants.BERT_IX_COLNAME} integer NOT NULL );"
    )
    # print(q)
    cursor.execute(q)
    inference_db_conn.commit()
    q = f"""INSERT INTO {constants.O2B_TABLE_NAME} 
                ({constants.SENTENCE_IX_COLNAME}, {constants.TOKEN_IX_COLNAME}, {constants.BERT_IX_COLNAME}) 
                VALUES (?, ?, ?)"""

    cursor.executemany(q, dfs_large)
    inference_db_conn.commit()
    sentence_ix2features_path = infer_on_sentences(
        dataloader, model, layer_nums, inference_db_conn
    )

    logger.debug(
        f"AFTER calc_infer() "
        f"model = {model} "
        f"tokenizer = {tokenizer} "
        f"dataset_to_infer_to = {dataset_to_infer_to} "
        f"layer_nums = {layer_nums} "
        f"is_word_level = {is_word_level} "
        f"number_of_sentences = {number_of_sentences} "
        f"path_for_db = {path_for_db} "
        f"sentence_ix2features_path = {sentence_ix2features_path} "
    )

    inference_db_conn.close()
    return db_path


def configure_dataloader(number_of_sentences, sentences_tokenization_df, tokenizer):
    logger.debug(
        f"BEFORE configure_dataloader() "
        f"number_of_sentences = {number_of_sentences} "
        f"sentences_tokenization_df_shape = {sentences_tokenization_df.shape} "
        f"tokenizer = {tokenizer} "
    )
    dataset = MorphoDataset.from_sentences_tokenization_df(
        sentences_tokenization_df, tokenizer.pad_token_id
    )
    num_of_sentences = len(sentences_tokenization_df)
    if number_of_sentences is None:
        number_of_sentences = num_of_sentences
    number_of_sentences = min(number_of_sentences, num_of_sentences)
    sentences_to_infer_ixes = random.sample(
        range(num_of_sentences), number_of_sentences
    )
    sentences_to_infer_dataset = torch.utils.data.Subset(
        dataset, sentences_to_infer_ixes
    )
    if sentences_to_infer_dataset:
        dataloader = torch.utils.data.DataLoader(
            sentences_to_infer_dataset, batch_size=32, shuffle=True
        )
    else:
        dataloader = []
    logger.debug(
        f"AFTER configure_dataloader() "
        f"number_of_sentences = {number_of_sentences} "
        f"sentences_tokenization_df_shape = {sentences_tokenization_df.shape} "
        f"tokenizer = {tokenizer} "
        f"dataloader_class = {type(dataloader)} "
    )
    return dataloader
