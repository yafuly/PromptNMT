# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import thumt.models.transformer


def get_model(name):
    name = name.lower()

    if name == "transformer":
        return thumt.models.transformer.Transformer
    elif name == "phase1transformer":
        return thumt.models.transformer.Phase1Transformer
    elif name == "phase1septransformer":
        return thumt.models.transformer.Phase1SepTransformer
    else:
        raise LookupError("Unknown model %s" % name)
