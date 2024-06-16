import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.nmslib_int import NmslibANN
from test_utils import verify_build_and_query, verify_save_and_load, data


class TestNmslibANN:
    @pytest.fixture(autouse=True)
    def setup_nmslib(self, data):
        self.ann = NmslibANN()
        self.data = data

    def test_build_and_query(self, data):
        verify_build_and_query(self.ann, self.data)

    def test_save_and_load_index(self, data):
        verify_save_and_load(self.ann, self.data, 'test_nmslib_index')
