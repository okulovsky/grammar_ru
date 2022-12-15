from typing import *
from math import ceil

import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import pandas as pd
import torch

from ....common.ml.batched_training.torch.networks.lstm_network import LSTMFinalizer
from ....common.ml.batched_training.torch.networks.network_commons import AnnotatedTensor


class LSTM_BicontextualFinalizer(LSTMFinalizer):
    """
        Applies a LSTM finalizer and changes the context so that 
        each sample has an element from both the left context and the right context.

        If LSTM finalizer output is [offset, n_samples, n_features], 
        then bicontextual finalizer output is [ceil(offset / 2), n_samples, n_features * 2].

        If offset is odd, then element with zero offser will be duplicated.

        Option mirror_concat allow change contexts concatenation direction.
    """
    def __init__(self, mirror_concat: bool = False) -> None:
        self.mirror_concat = mirror_concat
        super(LSTM_BicontextualFinalizer, self).__init__(reverse_context_order = False)

    def finalize(
        self, index: pd.DataFrame, 
        features: Dict[str, pd.DataFrame], aggregations: Dict[str, pd.DataFrame]):

        lstm_finalizer_result: AnnotatedTensor = super().finalize(index, features, aggregations)
        return self._bicontextual_transformation(lstm_finalizer_result)

    def _bicontextual_transformation(self, lstm_finalizer_result: AnnotatedTensor) -> AnnotatedTensor:
        
        offset_indices = lstm_finalizer_result.dim_indices[0]
        l, r = offset_indices[0], offset_indices[-1]
        c = offset_indices[len(offset_indices) // 2]
        offset_name = lstm_finalizer_result.dim_names[0]

        left_context = lstm_finalizer_result.sample_index(
            pd.Index(data = range(l, c, 1), name = offset_name)).tensor

        central_context = lstm_finalizer_result.sample_index(
            pd.Index(data = [c], name = offset_name)).tensor

        right_context = lstm_finalizer_result.sample_index(
            pd.Index(data = range(c + 1, r + 1), name = offset_name)).tensor

        if self.mirror_concat:
            right_context = torch.flip(right_context, dims = (0, ))

        fake_context = torch.Tensor()

        if left_context.shape == right_context.shape:
            
            result = self._concat(
                lhs = (left_context, right_context), rhs = (central_context, central_context), 
                inner_dim = 2, outer_dim = 0
            )
            
        elif left_context.shape >= right_context.shape:
            rhs = (central_context, right_context)
            if self.mirror_concat:
                rhs = list(reversed(rhs))
            result = self._concat(
                lhs = (left_context, fake_context), rhs = rhs,
                inner_dim = 0, outer_dim = 2
            )

        else:
            lhs = (left_context, central_context)
            if self.mirror_concat:
                lhs = list(reversed(lhs))
            result = self._concat(
                lhs = lhs, rhs = (right_context, fake_context),
                inner_dim = 0, outer_dim = 2
            )

        return AnnotatedTensor(
            tensor = result,
            dim_names = lstm_finalizer_result.dim_names,
            dim_indices = [
                offset_indices[:ceil(len(offset_indices)/2)],
                lstm_finalizer_result.dim_indices[1],
                lstm_finalizer_result.dim_indices[2] * 2
            ]
        )

    def _concat( self,
                lhs: Tuple[torch.Tensor], 
                rhs: Tuple[torch.Tensor], inner_dim: int, outer_dim: int) -> torch.Tensor:

        result = torch.cat(
            (
                torch.cat(lhs, dim = inner_dim), 
                torch.cat(rhs, dim = inner_dim)
            ),
            dim = outer_dim
        )

        return result
