import torch

import constants


class MorphoDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, tokens_ixes_to_remember, sentence_ids, pad_tok_ix):
        self.tokens = tokens
        self.tokens_ixes_to_remember = tokens_ixes_to_remember
        self.sentence_ids = sentence_ids
        self.pad_tok_ix = pad_tok_ix

    def __getitem__(self, item_ix):
        token_ids, attention_ids = bert_tokens_ixes2model_entry(
            self.tokens[item_ix], self.pad_tok_ix
        )
        token_ids_to_remember = torch.LongTensor(
            self.tokens_ixes_to_remember[item_ix]
            + [-1] * (512 - len(self.tokens_ixes_to_remember[item_ix]))
        )
        return (
            self.sentence_ids[item_ix],
            token_ids_to_remember,
            token_ids,
            attention_ids,
        )

    def __len__(self):
        return len(self.tokens)

    @staticmethod
    def from_sentences_tokenization_df(sentences_tokenization_df, pad_tok_ix):
        return MorphoDataset(
            sentences_tokenization_df.bert_tokenized_ixes,
            sentences_tokenization_df.bert_token_ixes_to_remember,
            sentences_tokenization_df[constants.SENTENCE_IX_COLNAME],
            pad_tok_ix,
        )


def bert_tokens_ixes2model_entry(tokens_ixes, pad_tok_ix):
    sentence = tokens_ixes[:512]
    token_ids = torch.LongTensor(sentence + [pad_tok_ix] * (512 - len(sentence))).cuda()
    attention_ids = torch.LongTensor(
        [1] * len(sentence) + [0] * (512 - len(sentence))
    ).cuda()
    return token_ids, attention_ids
