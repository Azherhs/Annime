import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../annime')))
from annime.scann_int import ScannANN
from test_utils import verify_build_and_query, verify_save_and_load, data


class TestScannANN:
    @pytest.fixture(autouse=True)
    def setup_scann(self, data):
        self.ann = ScannANN()
        self.data = data

    def test_build_and_query(self):
        verify_build_and_query(self.ann, self.data)

    def test_save_and_load_index(self):
        verify_save_and_load(self.ann, 'test_scann_index')
