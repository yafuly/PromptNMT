# coding=utf-8
# Copyright 2017-Present The THUMT Authors

import torch

from thumt.data.dataset import Dataset, ElementSpec, MapFunc, TextLineDataset
from thumt.data.vocab import Vocabulary
from thumt.tokenizers import WhiteSpaceTokenizer, AdviceTokenizer


def _sort_input_file(filename, keys=None, reverse=True):

    with open(filename, "rb") as fd:
        inputs = [line.strip() for line in fd]

    if keys is not None: # sort with keys
        assert len(keys) == len(inputs)
        sorted_inputs = [inputs[keys[i]] for i in range(len(keys))]
        return keys, sorted_inputs

    input_lens = [
        (i, len(line.split())) for i, line in enumerate(inputs)]

    sorted_input_lens = sorted(input_lens, key=lambda x: x[1],
                               reverse=reverse)
    sorted_keys = {}
    sorted_inputs = []

    for i, (idx, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[idx])
        sorted_keys[idx] = i

    return sorted_keys, sorted_inputs


class MTPipeline(object):

    @staticmethod
    def get_train_dataset(filenames, params, cpu=False):
        src_vocab = params.vocabulary["source"]
        tgt_vocab = params.vocabulary["target"]

        src_dataset = TextLineDataset(filenames[0])
        tgt_dataset = TextLineDataset(filenames[1])
        lab_dataset = TextLineDataset(filenames[1])

        src_dataset = src_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, params.eos)
        tgt_dataset = tgt_dataset.tokenize(WhiteSpaceTokenizer(),
                                           params.bos, None)
        lab_dataset = lab_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, params.eos)
        src_dataset = Dataset.lookup(src_dataset, src_vocab,
                                     src_vocab[params.unk])
        tgt_dataset = Dataset.lookup(tgt_dataset, tgt_vocab,
                                     tgt_vocab[params.unk])
        lab_dataset = Dataset.lookup(lab_dataset, tgt_vocab,
                                     tgt_vocab[params.unk])

        dataset = Dataset.zip((src_dataset, tgt_dataset, lab_dataset))
        dataset = dataset.shard(torch.distributed.get_world_size(),
                                torch.distributed.get_rank())


        def bucket_boundaries(max_length, min_length=8, step=8):
            x = min_length
            boundaries = []

            while x <= max_length:
                boundaries.append(x + 1)
                x += step

            return boundaries

        batch_size = params.batch_size
        max_length = (params.max_length // 8) * 8
        min_length = params.min_length
        boundaries = bucket_boundaries(max_length)
        batch_sizes = [max(1, batch_size // (x - 1))
                       if not params.fixed_batch_size else batch_size
                       for x in boundaries] + [1]

        dataset = Dataset.bucket_by_sequence_length(
            dataset, boundaries, batch_sizes, pad=src_vocab[params.pad],
            min_length=params.min_length, max_length=params.max_length)

        def map_fn(inputs):
            src_seq, tgt_seq, labels = inputs
            src_seq = torch.tensor(src_seq)
            tgt_seq = torch.tensor(tgt_seq)
            labels = torch.tensor(labels)
            src_mask = src_seq != params.vocabulary["source"][params.pad]
            tgt_mask = tgt_seq != params.vocabulary["target"][params.pad]
            src_mask = src_mask.float()
            tgt_mask = tgt_mask.float()

            if not cpu:
                src_seq = src_seq.cuda(params.device)
                src_mask = src_mask.cuda(params.device)
                tgt_seq = tgt_seq.cuda(params.device)
                tgt_mask = tgt_mask.cuda(params.device)

            features = {
                "source": src_seq,
                "source_mask": src_mask,
                "target": tgt_seq,
                "target_mask": tgt_mask
            }

            return features, labels

        map_obj = MapFunc(map_fn, ElementSpec("Tensor", "{key: [None, None]}"))

        dataset = dataset.map(map_obj)
        dataset = dataset.background()

        return dataset

    @staticmethod
    def get_eval_dataset(filenames, params, cpu=False):
        src_vocab = params.vocabulary["source"]
        tgt_vocab = params.vocabulary["target"]

        src_dataset = TextLineDataset(filenames[0])
        tgt_dataset = TextLineDataset(filenames[1])
        lab_dataset = TextLineDataset(filenames[1])

        src_dataset = src_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, params.eos)
        tgt_dataset = tgt_dataset.tokenize(WhiteSpaceTokenizer(),
                                           params.bos, None)
        lab_dataset = lab_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, params.eos)
        src_dataset = Dataset.lookup(src_dataset, src_vocab,
                                     src_vocab[params.unk])
        tgt_dataset = Dataset.lookup(tgt_dataset, tgt_vocab,
                                     tgt_vocab[params.unk])
        lab_dataset = Dataset.lookup(lab_dataset, tgt_vocab,
                                     tgt_vocab[params.unk])

        dataset = Dataset.zip((src_dataset, tgt_dataset, lab_dataset))
        dataset = dataset.shard(torch.distributed.get_world_size(),
                                torch.distributed.get_rank())

        dataset = dataset.padded_batch(params.decode_batch_size,
                                       pad=src_vocab[params.pad])

        def map_fn(inputs):
            src_seq, tgt_seq, labels = inputs
            src_seq = torch.tensor(src_seq)
            tgt_seq = torch.tensor(tgt_seq)
            labels = torch.tensor(labels)
            src_mask = src_seq != params.vocabulary["source"][params.pad]
            tgt_mask = tgt_seq != params.vocabulary["target"][params.pad]
            src_mask = src_mask.float()
            tgt_mask = tgt_mask.float()

            if not cpu:
                src_seq = src_seq.cuda(params.device)
                src_mask = src_mask.cuda(params.device)
                tgt_seq = tgt_seq.cuda(params.device)
                tgt_mask = tgt_mask.cuda(params.device)

            features = {
                "source": src_seq,
                "source_mask": src_mask,
                "target": tgt_seq,
                "target_mask": tgt_mask
            }

            return features, labels

        map_obj = MapFunc(map_fn, ElementSpec("Tensor", "{key: [None, None]}"))

        dataset = dataset.map(map_obj)
        dataset = dataset.background()

        return dataset

    @staticmethod
    def get_infer_dataset(filename, params, cpu=False):
        sorted_keys, sorted_data = _sort_input_file(filename)
        src_vocab = params.vocabulary["source"]

        src_dataset = TextLineDataset(sorted_data)
        src_dataset = src_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, params.eos)
        src_dataset = Dataset.lookup(src_dataset, src_vocab,
                                     src_vocab[params.unk])
        dataset = src_dataset.shard(torch.distributed.get_world_size(),
                                    torch.distributed.get_rank())

        dataset = dataset.padded_batch(params.decode_batch_size,
                                       pad=src_vocab[params.pad])

        def map_fn(inputs):
            src_seq = torch.tensor(inputs)
            src_mask = src_seq != params.vocabulary["source"][params.pad]
            src_mask = src_mask.float()

            if not cpu:
                src_seq = src_seq.cuda(params.device)
                src_mask = src_mask.cuda(params.device)

            features = {
                "source": src_seq,
                "source_mask": src_mask,
            }

            return features

        map_obj = MapFunc(map_fn, ElementSpec("Tensor", "{key: [None, None]}"))

        dataset = dataset.map(map_obj)
        dataset = dataset.background()

        return sorted_keys, dataset


class MTIS1Pipeline(MTPipeline):
    @staticmethod
    def get_train_dataset(filenames, params, cpu=False):
        src_vocab = params.vocabulary["source"]
        tgt_vocab = params.vocabulary["target"]

        src_dataset = TextLineDataset(filenames[0])
        tgt_dataset = TextLineDataset(filenames[1])
        lab_dataset = TextLineDataset(filenames[1])
        advice_dataset = TextLineDataset(filenames[2])

        src_dataset = src_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, params.eos)
        tgt_dataset = tgt_dataset.tokenize(WhiteSpaceTokenizer(),
                                           params.bos, None)
        lab_dataset = lab_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, params.eos)
        advice_dataset = advice_dataset.tokenize(AdviceTokenizer(ratio=params.ratio, flip_ratio=params.flip_ratio, sample_mode=params.sample_train_mode, crl_factor=params.crl_factor, formalize=params.formalize),
                                           None, params.eos, params.seed) # sample and shuffle advices

        src_dataset = Dataset.lookup(src_dataset, src_vocab,
                                     src_vocab[params.unk])
        tgt_dataset = Dataset.lookup(tgt_dataset, tgt_vocab,
                                     tgt_vocab[params.unk])
        lab_dataset = Dataset.lookup(lab_dataset, tgt_vocab,
                                     tgt_vocab[params.unk])
        advice_dataset = Dataset.lookup(advice_dataset, src_vocab,
                                     src_vocab[params.unk])

        dataset = Dataset.zip((src_dataset, tgt_dataset, lab_dataset, advice_dataset))
        dataset = dataset.shard(torch.distributed.get_world_size(),
                                torch.distributed.get_rank())


        def bucket_boundaries(max_length, min_length=8, step=8):
            x = min_length
            boundaries = []

            while x <= max_length:
                boundaries.append(x + 1)
                x += step

            return boundaries

        batch_size = params.batch_size
        max_length = (params.max_length // 8) * 8
        min_length = params.min_length
        boundaries = bucket_boundaries(max_length)
        batch_sizes = [max(1, batch_size // (x - 1))
                       if not params.fixed_batch_size else batch_size
                       for x in boundaries] + [1]

        dataset = Dataset.bucket_by_sequence_length(
            dataset, boundaries, batch_sizes, pad=src_vocab[params.pad],
            min_length=params.min_length, max_length=params.max_length)

        def map_fn(inputs):
            src_seq, tgt_seq, labels, advice = inputs
            src_seq = torch.tensor(src_seq)
            tgt_seq = torch.tensor(tgt_seq)
            labels = torch.tensor(labels)
            advice = torch.tensor(advice)

            src_mask = src_seq != params.vocabulary["source"][params.pad]
            tgt_mask = tgt_seq != params.vocabulary["target"][params.pad]
            advice_mask = advice != params.vocabulary["source"][params.pad] 
            src_mask = src_mask.float()
            tgt_mask = tgt_mask.float()
            advice_mask = advice_mask.float()

            if not cpu:
                src_seq = src_seq.cuda(params.device)
                src_mask = src_mask.cuda(params.device)
                tgt_seq = tgt_seq.cuda(params.device)
                tgt_mask = tgt_mask.cuda(params.device)
                advice = advice.cuda(params.device)
                advice_mask = advice_mask.cuda(params.device)

            features = {
                "source": src_seq,
                "source_mask": src_mask,
                "target": tgt_seq,
                "target_mask": tgt_mask,
                "advice": advice,
                "advice_mask": advice_mask
            }

            return features, labels

        map_obj = MapFunc(map_fn, ElementSpec("Tensor", "{key: [None, None]}"))

        dataset = dataset.map(map_obj)
        dataset = dataset.background()

        return dataset


    @staticmethod
    def get_eval_dataset(filenames, params, cpu=False):
        raise NotImplementedError

    @staticmethod
    def get_infer_dataset(filename, params, cpu=False):
        data, advice = filename.split("\t")
        sorted_keys, sorted_data = _sort_input_file(data)
        reverse_keys = {v:k for k,v in sorted_keys.items()}
        _, sorted_advice = _sort_input_file(advice, keys=reverse_keys)
        # reverse_keys = {v:k for k,v in sorted_keys.items()}
        # sorted_advice = [advice[reverse_keys[i]] for i in range(len(advice))]

        src_vocab = params.vocabulary["source"]

        # print(params.seed)
        # print(params.ratio)
        src_dataset = TextLineDataset(sorted_data)
        advice_dataset = TextLineDataset(sorted_advice)
        src_dataset = src_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, params.eos)
        advice_dataset = advice_dataset.tokenize(AdviceTokenizer(ratio=params.ratio, flip_ratio=params.flip_ratio, sample_mode=params.sample_infer_mode, formalize=params.formalize),
                                            None, params.eos, params.seed)
                                        
        src_dataset = Dataset.lookup(src_dataset, src_vocab,
                                     src_vocab[params.unk])
        advice_dataset = Dataset.lookup(advice_dataset, src_vocab,
                                        src_vocab[params.unk])



        dataset = Dataset.zip((src_dataset, advice_dataset))                       
        dataset = dataset.shard(torch.distributed.get_world_size(),
                                    torch.distributed.get_rank())

        dataset = dataset.padded_batch(params.decode_batch_size,
                                       pad=src_vocab[params.pad])

        def map_fn(inputs):
            src_seq, advice = inputs

            src_seq = torch.tensor(src_seq)
            advice = torch.tensor(advice)

            src_mask = src_seq != params.vocabulary["source"][params.pad]
            src_mask = src_mask.float()
            advice_mask = advice != params.vocabulary["source"][params.pad]
            advice_mask = advice_mask.float()

            if not cpu:
                src_seq = src_seq.cuda(params.device)
                src_mask = src_mask.cuda(params.device)
                advice = advice.cuda(params.device)
                advice_mask = advice_mask.cuda(params.device)

            features = {
                "source": src_seq,
                "source_mask": src_mask,
                "advice": advice,
                "advice_mask": advice_mask
            }

            return features

        map_obj = MapFunc(map_fn, ElementSpec("Tensor", "{key: [None, None]}"))

        dataset = dataset.map(map_obj)
        dataset = dataset.background()

        return sorted_keys, dataset

class MTIS2Pipeline(object):

    @staticmethod
    def get_train_dataset(filenames, params, cpu=False):
        src_vocab = params.vocabulary["source"]
        tgt_vocab = params.vocabulary["target"]

        src_dataset = TextLineDataset(filenames[0])
        tgt_dataset = TextLineDataset(filenames[1])
        lab_dataset = TextLineDataset(filenames[1])

        src_dataset = src_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, params.eos)
        tgt_dataset = tgt_dataset.tokenize(WhiteSpaceTokenizer(),
                                           params.bos, None)
        lab_dataset = lab_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, params.eos)
        src_dataset = Dataset.lookup(src_dataset, src_vocab,
                                     src_vocab[params.unk])
        tgt_dataset = Dataset.lookup(tgt_dataset, tgt_vocab,
                                     tgt_vocab[params.unk])
        lab_dataset = Dataset.lookup(lab_dataset, tgt_vocab,
                                     tgt_vocab[params.unk])

        dataset = Dataset.zip((src_dataset, tgt_dataset, lab_dataset))
        dataset = dataset.shard(torch.distributed.get_world_size(),
                                torch.distributed.get_rank())


        def bucket_boundaries(max_length, min_length=8, step=8):
            x = min_length
            boundaries = []

            while x <= max_length:
                boundaries.append(x + 1)
                x += step

            return boundaries

        batch_size = params.batch_size
        max_length = (params.max_length // 8) * 8
        min_length = params.min_length
        boundaries = bucket_boundaries(max_length)
        batch_sizes = [max(1, batch_size // (x - 1))
                       if not params.fixed_batch_size else batch_size
                       for x in boundaries] + [1]

        dataset = Dataset.bucket_by_sequence_length(
            dataset, boundaries, batch_sizes, pad=src_vocab[params.pad],
            min_length=params.min_length, max_length=params.max_length)

        def map_fn(inputs):
            src_seq, tgt_seq, labels = inputs
            src_seq = torch.tensor(src_seq)
            tgt_seq = torch.tensor(tgt_seq)
            labels = torch.tensor(labels)
            src_mask = src_seq != params.vocabulary["source"][params.pad]
            tgt_mask = tgt_seq != params.vocabulary["target"][params.pad]
            src_mask = src_mask.float()
            tgt_mask = tgt_mask.float()

            if not cpu:
                src_seq = src_seq.cuda(params.device)
                src_mask = src_mask.cuda(params.device)
                tgt_seq = tgt_seq.cuda(params.device)
                tgt_mask = tgt_mask.cuda(params.device)

            features = {
                "source": src_seq,
                "source_mask": src_mask,
                "target": tgt_seq,
                "target_mask": tgt_mask
            }
            print(features["target"])
            print(features["target_mask"])
            xxx

            return features, labels

        map_obj = MapFunc(map_fn, ElementSpec("Tensor", "{key: [None, None]}"))

        dataset = dataset.map(map_obj)
        dataset = dataset.background()

        return dataset

    @staticmethod
    def get_eval_dataset(filenames, params, cpu=False):
        src_vocab = params.vocabulary["source"]
        tgt_vocab = params.vocabulary["target"]

        src_dataset = TextLineDataset(filenames[0])
        tgt_dataset = TextLineDataset(filenames[1])
        lab_dataset = TextLineDataset(filenames[1])

        src_dataset = src_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, params.eos)
        tgt_dataset = tgt_dataset.tokenize(WhiteSpaceTokenizer(),
                                           params.bos, None)
        lab_dataset = lab_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, params.eos)
        src_dataset = Dataset.lookup(src_dataset, src_vocab,
                                     src_vocab[params.unk])
        tgt_dataset = Dataset.lookup(tgt_dataset, tgt_vocab,
                                     tgt_vocab[params.unk])
        lab_dataset = Dataset.lookup(lab_dataset, tgt_vocab,
                                     tgt_vocab[params.unk])

        dataset = Dataset.zip((src_dataset, tgt_dataset, lab_dataset))
        dataset = dataset.shard(torch.distributed.get_world_size(),
                                torch.distributed.get_rank())

        dataset = dataset.padded_batch(params.decode_batch_size,
                                       pad=src_vocab[params.pad])

        def map_fn(inputs):
            src_seq, tgt_seq, labels = inputs
            src_seq = torch.tensor(src_seq)
            tgt_seq = torch.tensor(tgt_seq)
            labels = torch.tensor(labels)
            src_mask = src_seq != params.vocabulary["source"][params.pad]
            tgt_mask = tgt_seq != params.vocabulary["target"][params.pad]
            src_mask = src_mask.float()
            tgt_mask = tgt_mask.float()

            if not cpu:
                src_seq = src_seq.cuda(params.device)
                src_mask = src_mask.cuda(params.device)
                tgt_seq = tgt_seq.cuda(params.device)
                tgt_mask = tgt_mask.cuda(params.device)

            features = {
                "source": src_seq,
                "source_mask": src_mask,
                "target": tgt_seq,
                "target_mask": tgt_mask
            }

            return features, labels

        map_obj = MapFunc(map_fn, ElementSpec("Tensor", "{key: [None, None]}"))

        dataset = dataset.map(map_obj)
        dataset = dataset.background()

        return dataset

    @staticmethod
    def get_infer_dataset(filename, params, cpu=False):
        sorted_keys, sorted_data = _sort_input_file(filename)
        src_vocab = params.vocabulary["source"]

        src_dataset = TextLineDataset(sorted_data)
        src_dataset = src_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, params.eos)
        src_dataset = Dataset.lookup(src_dataset, src_vocab,
                                     src_vocab[params.unk])
        dataset = src_dataset.shard(torch.distributed.get_world_size(),
                                    torch.distributed.get_rank())

        dataset = dataset.padded_batch(params.decode_batch_size,
                                       pad=src_vocab[params.pad])

        def map_fn(inputs):
            src_seq = torch.tensor(inputs)
            src_mask = src_seq != params.vocabulary["source"][params.pad]
            src_mask = src_mask.float()

            if not cpu:
                src_seq = src_seq.cuda(params.device)
                src_mask = src_mask.cuda(params.device)

            features = {
                "source": src_seq,
                "source_mask": src_mask,
            }

            return features

        map_obj = MapFunc(map_fn, ElementSpec("Tensor", "{key: [None, None]}"))

        dataset = dataset.map(map_obj)
        dataset = dataset.background()

        return sorted_keys, dataset
