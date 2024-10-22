import pathlib
import torch.nn as nn
import torchdata.datapipes as dp
from torch.hub import load_state_dict_from_url

import torchtext.transforms as T
from evaluation.torch_utils import TorchEvalSpec
from torch.utils.data import DataLoader

DATASET_LOC = "datasets/wikitext103"


def build_datapipe(root, spec: TorchEvalSpec):
    encoder_json_path = (
        "https://download.pytorch.org/models/text/gpt2_bpe_encoder.json"
    )
    vocab_bpe_path = (
        "https://download.pytorch.org/models/text/gpt2_bpe_vocab.bpe"
    )
    tokenizer = T.GPT2BPETokenizer(encoder_json_path, vocab_bpe_path)
    vocab_path = "https://download.pytorch.org/models/text/roberta.vocab.pt"
    vocab = T.VocabTransform(load_state_dict_from_url(vocab_path))
    add_bos = T.AddToken(token=0, begin=True)
    add_eos = T.AddToken(token=2, begin=False)

    embedding = nn.Embedding(50257, 764, _freeze=True)

    datapipe = dp.iter.FileLister(root=root, recursive=True)
    datapipe = dp.iter.FileOpener(datapipe)
    datapipe = dp.iter.LineReader(datapipe, return_path=False)
    datapipe = datapipe.sharding_filter()

    datapipe = dp.iter.Mapper(datapipe, tokenizer)
    datapipe = dp.iter.Mapper(datapipe, T.Truncate(max_seq_len=254))
    datapipe = dp.iter.Mapper(datapipe, vocab)
    datapipe = dp.iter.Mapper(datapipe, add_bos)
    datapipe = dp.iter.Mapper(datapipe, add_eos)
    datapipe = dp.iter.Mapper(datapipe, T.ToTensor())
    datapipe = dp.iter.Mapper(datapipe, embedding)

    return datapipe


def get_dataset(spec: TorchEvalSpec):
    data_dir = (
        pathlib.Path(__file__).resolve().parents[2].joinpath(DATASET_LOC)
    )
    train_filepath = pathlib.Path(data_dir) / pathlib.Path(
        "wikitext-103/wiki.train.tokens"
    )

    datapipe = build_datapipe(str(train_filepath), spec)

    dataloader = DataLoader(
        datapipe, batch_size=spec.batch_size, num_workers=spec.num_workers
    )

    return dataloader


if __name__ == "__main__":
    dataset = get_dataset(TorchEvalSpec(1, 1))
    for i, x in enumerate(dataset):
        print(x)
        if i == 10:
            break
