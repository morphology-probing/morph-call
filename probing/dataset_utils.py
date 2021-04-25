import logging

import pandas as pd

import constants

logger = logging.getLogger(f"{constants.LOGGER_NAME}.dataset_utils")


def create_dataset(
    fname: str,
    word_colname: str = constants.WORD_COLNAME,
    train_size: int = None,
    test_size: int = None,
    val_size: int = None,
    mask_token: str = None,
):
    logger.debug(f"BEFORE create_dataset() fname = {fname}")

    probing_dataset = pd.read_csv(fname, delimiter="\t")
    word_level_probing = word_colname in probing_dataset.columns

    probing_dataset[constants.TOKENIZED_COLNAME] = probing_dataset[
        constants.SENTENCE_TEXT_COLNAME
    ].apply(lambda x: x.split())
    if not word_level_probing:
        probing_dataset[constants.TOKEN_IX_COLNAME] = constants.FAKE_TOK_IX
        probing_dataset[constants.DUPLICATE_SENTENCE_TEXT_COLNAME] = probing_dataset[
            constants.SENTENCE_TEXT_COLNAME
        ]

    probing_dataset[constants.PROBING_ITEM_IX_COLNAME] = probing_dataset.index
    probing_dataset[constants.SENTENCE_IX_COLNAME] = probing_dataset.groupby(
        by=constants.SENTENCE_TEXT_COLNAME
    ).ngroup()

    max_sent_id = probing_dataset[constants.SENTENCE_IX_COLNAME].max()

    train_dataset = probing_dataset[
        probing_dataset[constants.SUBSAMPLE_COLNAME] == constants.TRAIN_SUBSAMPLE_NAME
    ]
    test_dataset = probing_dataset[
        probing_dataset[constants.SUBSAMPLE_COLNAME] == constants.TEST_SUBSAMPLE_NAME
    ]

    if word_level_probing and mask_token is not None:
        test_dataset[word_colname] = mask_token

        def splitted_masker(row):
            splitted = row[constants.TOKENIZED_COLNAME]
            splitted[row[constants.TOKEN_IX_COLNAME]] = mask_token
            return splitted

        test_dataset[constants.TOKENIZED_COLNAME] = test_dataset.apply(
            splitted_masker, axis=1
        )
        test_dataset[constants.SENTENCE_IX_COLNAME] = range(
            max_sent_id + 1, max_sent_id + 1 + len(test_dataset)
        )

    val_dataset = probing_dataset[
        probing_dataset[constants.SUBSAMPLE_COLNAME] == constants.VAL_SUBSAMPLE_NAME
    ]

    if train_size is not None and train_size < train_dataset.shape[0]:
        train_dataset = train_dataset.sample(n=train_size)
        logger.info(
            f"training probing dataset reduced to {train_dataset.shape[0]} size"
        )
    if test_size is not None and test_size < test_dataset.shape[0]:
        test_dataset = test_dataset.sample(n=test_size)
        logger.info(f"test probing dataset reduced to {test_dataset.shape[0]} size")
    if val_size is not None and val_size < val_dataset.shape[0]:
        val_dataset = val_dataset.sample(n=val_size)
        logger.info(
            f"validation probing dataset reduced to {val_dataset.shape[0]} size"
        )

    dfs_names = ["train", "val", "test"]
    dfs = [train_dataset, val_dataset, test_dataset]

    logger.debug(
        f"AFTER create_dataset() fname = {fname} "
        f"created {len(dfs)} dataframes ( {dfs_names} ) "
        f"with shapes {[df.shape for df in dfs]}. word_level_probing={word_level_probing}"
    )
    return dfs_names, dfs, word_level_probing
