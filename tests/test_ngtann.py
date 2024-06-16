import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.ngt_int import NgtANN
from test_utils import verify_build_and_query, verify_save_and_load, data


class TestNgtANN:
    @pytest.fixture(autouse=True)
    def setup_ngt(self, data):
        self.ann = NgtANN(dim=10)
        self.data = data

    def test_build_and_query(self, data):
        verify_build_and_query(self.ann, self.data)

    def test_save_and_load_index(self, data):
        self.ann.build_index(self.data)
        verify_save_and_load(self.ann, 'test_ngt_index')
