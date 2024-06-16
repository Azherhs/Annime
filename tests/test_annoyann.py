import pytest
import numpy as np
import os
import sys
from test_utils import verify_build_and_query, verify_save_and_load

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.annoy_int import AnnoyANN


@pytest.fixture(scope='module')
def data():
    return np.random.rand(100, 10)


class TestAnnoyANN:
    @pytest.fixture(autouse=True)
    def setup_annoy(self, data):
        self.ann = AnnoyANN(dim=10)
        self.data = data

    def test_build_and_query(self):
        verify_build_and_query(self.ann, self.data)

    def test_save_and_load_index(self):
        self.ann.build_index(self.data)
        verify_save_and_load(self.ann, 'test_annoy_index.ann')
