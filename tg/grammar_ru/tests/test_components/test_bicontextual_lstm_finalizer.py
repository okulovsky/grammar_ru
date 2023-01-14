from tg.grammar_ru.components.bicontextual_lstm_finalizer import LSTM_BicontextualFinalizer
from tg.common.ml.batched_training.torch import AnnotatedTensor

from unittest import TestCase
import torch


class LSTM_BicontextualFinalizerTestCase(TestCase):
    def setUp(self) -> None:
        self.dim_names = ["context", "sample_id", "features"]

        self.odd_annotated_tensor = AnnotatedTensor(
            tensor = torch.Tensor(
                [[[0.1, 0.4], [0.7, 0.5]],
                 [[0.7, 0.3], [0.6, 0.4]],
                 [[0.4, 0.4], [0.5, 0.4]]
                ]
            ),
            dim_names = self.dim_names,
            dim_indices = [list(range(-1,2)), [0, 1], ["f1", "f2"]]
        )

        self.even_annotated_tensor = AnnotatedTensor(
            tensor = torch.Tensor(
                [[[0.1, 0.4], [0.7, 0.5]],
                 [[0.7, 0.3], [0.6, 0.4]],
                 [[0.4, 0.4], [0.5, 0.4]],
                 [[0.2, 0.8], [0.1, 0.9]]
                ]
            ),
            dim_names = self.dim_names,
            dim_indices = [list(range(-2,2)), [0, 1], ["f1", "f2"]]
        )

    def check(self, expected: AnnotatedTensor, result: AnnotatedTensor):
        self.assertTrue(torch.equal(expected.tensor, result.tensor))
        self.assertEqual(result.dim_names, expected.dim_names)
        self.assertEqual(result.dim_indices, expected.dim_indices)
    
    def test_odd_context_size(self):
        finalizer = LSTM_BicontextualFinalizer()
        result = finalizer._bicontextual_transformation(self.odd_annotated_tensor)
        
        expected = AnnotatedTensor(
            tensor = torch.Tensor(
                [[[0.1, 0.4, 0.4, 0.4], [0.7, 0.5, 0.5, 0.4]],
                 [[0.7, 0.3, 0.7, 0.3], [0.6, 0.4, 0.6, 0.4]],
                ]
            ),
            dim_names = self.dim_names,
            dim_indices = [[-1, 0], [0, 1], ["f1", "f2"] * 2]
        )

        self.check(expected, result)
    
    def test_even_context_size(self):
        finalizer = LSTM_BicontextualFinalizer()
        result = finalizer._bicontextual_transformation(self.even_annotated_tensor)

        expected = AnnotatedTensor(
            tensor = torch.Tensor(
                [[[0.1, 0.4, 0.4, 0.4], [0.7, 0.5, 0.5, 0.4]],
                 [[0.7, 0.3, 0.2, 0.8], [0.6, 0.4, 0.1, 0.9]]
                ]
            ),
            dim_names = self.dim_names,
            dim_indices = [[-2, -1], [0, 1], ["f1", "f2"] * 2]
        )

        self.check(expected, result)

    def test_mirror_concat(self):
        finalizer = LSTM_BicontextualFinalizer(mirror_concat = True)
        result = finalizer._bicontextual_transformation(self.even_annotated_tensor)

        expected = AnnotatedTensor(
            tensor = torch.Tensor(
                [[[0.1, 0.4, 0.2, 0.8], [0.7, 0.5, 0.1, 0.9]],
                 [[0.7, 0.3, 0.4, 0.4], [0.6, 0.4, 0.5, 0.4]]
                ]
            ),
            dim_names = self.dim_names,
            dim_indices = [[-2, -1], [0, 1], ["f1", "f2"] * 2]
        )

        self.check(expected, result)
