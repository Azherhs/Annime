import pytest
import numpy as np
import os
import sys
from sklearn.datasets import fetch_openml
from test_utils import verify_build_and_query, verify_save_and_load

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.annoy_int import AnnoyANN
from src.ngt_int import NgtANN
from src.nmslib_int import NmslibANN
from src.scann_int import ScannANN
from src.hnswlib_int import HnswlibANN
from src.faiss_int import FaissANN
from src.datasketch_ann import DatasketchANN

DATA_DIR = 'data'


@pytest.fixture(scope='module')
def mnist_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    mnist = fetch_openml('mnist_784', version=1, data_home=DATA_DIR)
    return mnist.data[:1000].astype(np.float32) / 255.0  # Normalize the data


class TestRealData:
    @pytest.fixture(autouse=True)
    def setup_mnist(self, mnist_data):
        self.mnist_data = mnist_data

    def test_annoy_mnist(self):
        ann = AnnoyANN(dim=784)
        verify_build_and_query(ann, self.mnist_data)
        verify_save_and_load(ann, self.mnist_data, 'test_annoy_mnist_index.ann')

    def test_ngt_mnist(self):
        ann = NgtANN(dim=784)
        verify_build_and_query(ann, self.mnist_data)
        verify_save_and_load(ann, self.mnist_data, 'test_ngt_mnist_index')

    def test_nmslib_mnist(self):
        ann = NmslibANN()
        verify_build_and_query(ann, self.mnist_data)
        verify_save_and_load(ann, self.mnist_data, 'test_nmslib_mnist_index')

    def test_scann_mnist(self):
        ann = ScannANN()
        verify_build_and_query(ann, self.mnist_data)
        verify_save_and_load(ann, self.mnist_data, 'test_scann_mnist_index')

    def test_hnswlib_mnist(self):
        ann = HnswlibANN(dim=784)
        verify_build_and_query(ann, self.mnist_data)
        verify_save_and_load(ann, self.mnist_data, 'test_hnswlib_mnist_index')

    def test_faiss_mnist(self):
        ann = FaissANN(dim=784)
        verify_build_and_query(ann, self.mnist_data)
        verify_save_and_load(ann, self.mnist_data, 'test_faiss_mnist_index')

    def test_datasketch_mnist(self):
        ann = DatasketchANN(num_perm=128, threshold=0.5)
        verify_build_and_query(ann, self.mnist_data)
        verify_save_and_load(ann, self.mnist_data, 'test_datasketch_mnist_index.pkl')
