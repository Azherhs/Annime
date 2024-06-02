import pytest
import numpy as np
import sys
import os
# Ensure the 'src' directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.annoy_int import AnnoyANN
from src.ngt_int import NgtANN
from src.nmslib_int import NmslibANN
from src.scann_int import ScannANN



@pytest.fixture(scope='module')
def data():
    return np.random.rand(100, 10)


class TestAnnoyANN:

    @pytest.fixture(autouse=True)
    def setup_annoy(self, data):
        self.ann = AnnoyANN(dim=10)
        self.ann.build_index(data)
        self.data = data

    def test_build_index(self):
        assert self.ann.built
        assert self.ann.get_index_size() == 100

    def test_query(self):
        query_point = self.data[0]
        result = self.ann.query(query_point, k=5)
        assert len(result[0]) == 5

    def test_save_load_index(self):
        self.ann.save_index('test_annoy_index.ann')
        self.ann.load_index('test_annoy_index.ann')
        assert self.ann.built


class TestNgtANN:

    @pytest.fixture(autouse=True)
    def setup_ngt(self, data):
        self.ann = NgtANN(dim=10)
        self.ann.build_index(data)
        self.data = data

    def test_build_index(self):
        assert self.ann.built
        assert self.ann.get_index_size() == 100

    def test_query(self):
        query_point = self.data[0]
        result = self.ann.query(query_point, k=5)
        assert len(result) == 5

    def test_save_load_index(self):
        self.ann.save_index('test_ngt_index')
        self.ann.load_index('test_ngt_index')
        assert self.ann.built


class TestNmslibANN:

    @pytest.fixture(autouse=True)
    def setup_nmslib(self, data):
        self.ann = NmslibANN()
        self.ann.build_index(data)
        self.data = data

    def test_build_index(self):
        assert self.ann.built
        assert self.ann.get_index_size() == 100

    def test_query(self):
        query_point = self.data[0]
        result = self.ann.query(query_point, k=5)
        assert len(result) == 5

    def test_save_load_index(self):
        self.ann.save_index('test_nmslib_index')
        self.ann.load_index('test_nmslib_index')
        assert self.ann.built


class TestScannANN:

    @pytest.fixture(autouse=True)
    def setup_scann(self, data):
        self.ann = ScannANN()
        self.ann.build_index(data)
        self.data = data

    def test_build_index(self):
        assert self.ann.index is not None
        assert self.ann.get_index_size() == 100

    def test_query(self):
        query_point = self.data[0]
        result = self.ann.query(query_point, k=5)
        assert len(result) == 5

    def test_save_load_index(self):
        self.ann.save_index('test_scann_index')
        self.ann.load_index('test_scann_index')
        assert self.ann.index is not None
