import numpy as np
import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../annime')))
from annime.faiss_int import FaissANN
from test_utils import verify_build_and_query, verify_save_and_load


@pytest.fixture(scope='module')
def data():
    return np.random.rand(100, 10).astype(str)


class TestFaissANN:
    @pytest.fixture(autouse=True)
    def setup_faiss(self, data):
        self.ann = FaissANN(dim=10)
        self.data = data

    def test_build_and_query(self, data):
        verify_build_and_query(self.ann, self.data)

    def test_save_and_load_index(self):
        verify_save_and_load(self.ann, self.data, 'test_faiss_index')
