from typing import Iterable

import pytest

from mock_data import MockDataset
from mock_data.backends import AbstractBackendInterface


def test_presense_of_core_data_backends():
    core_backends = [
        "BoundedNumerical",
        "BoundedDatetime",
        "WeightedDiscrete",
        "LoremIpsumText",
    ]

    for backend in core_backends:
        assert backend in MockDataset.BACKENDS


def test_registration_of_backend_fails_if_backend_is_not_subclass_of_interface():
    class NotGonnaWork:
        pass

    with pytest.raises(TypeError):
        MockDataset.register_backend(NotGonnaWork)


def test_registration_of_valid_backend_is_successful():
    # this is a backend that just "samples" ones. Valid nonetheless
    class OneGeneratorBackend(AbstractBackendInterface):
        def generate_samples(self, size: int) -> Iterable:
            return [1] * size

    MockDataset.register_backend(OneGeneratorBackend)
    assert "OneGeneratorBackend" in MockDataset.BACKENDS


# TODO: the generate_mock_data method and init must still be tested
def test_generation_of_mock_data():
    pass
