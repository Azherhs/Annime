import pytest
import numpy as np
import os
import sys
from test_utils import verify_build_and_query, verify_save_and_load

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.datasketch_ann import DatasketchANN


@pytest.fixture(scope='module')
def data():
    return np.random.rand(100, 10).astype(str)


class TestDatasketchANN:
    @pytest.fixture(autouse=True)
    def setup_datasketch(self, data):
        self.ann = DatasketchANN()
        self.data = data

    def test_build_and_query(self):
        verify_build_and_query(self.ann, self.data)

    def test_save_and_load_index(self):
        verify_save_and_load(self.ann, self.data, 'test_datasketch_index.pkl')
