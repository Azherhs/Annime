import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../annime')))
from annime.hnswlib_int import HnswlibANN
from test_utils import verify_build_and_query, verify_save_and_load, data


class TestHnswlibANN:
    @pytest.fixture(autouse=True)
    def setup_hnswlib(self, data):
        self.ann = HnswlibANN(dim=10)
        self.data = data

    def test_build_and_query(self, data):
        verify_build_and_query(self.ann, self.data)

    def test_save_and_load_index(self, data):
        # self.ann.build_index(self.data)
        verify_save_and_load(self.ann, self.data, 'test_hnswlib_index')
