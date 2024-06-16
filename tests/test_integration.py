import pytest
import numpy as np
import os
import sys
import time
from test_utils import verify_build_and_query, verify_save_and_load

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.annoy_int import AnnoyANN
from src.ngt_int import NgtANN
from src.nmslib_int import NmslibANN
from src.scann_int import ScannANN
from src.hnswlib_int import HnswlibANN
from src.faiss_int import FaissANN
from src.datasketch_ann import DatasketchANN


@pytest.fixture(scope='module')
def data():
    return np.random.rand(100, 10)


@pytest.fixture(scope='module')
def large_data():
    return np.random.rand(10000, 10)


@pytest.fixture(scope='module')
def query_points():
    return np.random.rand(10, 10)


class TestIntegration:
    @pytest.fixture(autouse=True)
    def setup_data(self, data):
        self.data = data

    def test_integration_of_interfaces(self):
        anns = [
            AnnoyANN(dim=10),
            NgtANN(dim=10),
            NmslibANN(),
            ScannANN(),
            HnswlibANN(dim=10),
            FaissANN(dim=10),
            DatasketchANN(num_perm=128, threshold=0.5)
        ]
        for ann in anns:
            verify_build_and_query(ann, self.data)
            verify_save_and_load(ann, self.data, f'test_{ann.__class__.__name__}_index')

    def test_combined_scenario(self):
        ann = HnswlibANN(dim=10)
        verify_build_and_query(ann, self.data)
        new_data = np.random.rand(10, 10)
        ann.add_items(new_data)
        assert ann.get_index_size() == 110
        query_point = self.data[0]
        result = ann.query(query_point, k=5)
        assert len(result[0]) == 5
        ann.save_index('test_combined_index')
        ann.load_index('test_combined_index')
        assert ann.built

    def test_search_time(self, large_data, query_points):
        ann = FaissANN(dim=10)
        ann.build_index(large_data)
        assert ann.built
        start_time = time.time()
        for query_point in query_points:
            ann.query(query_point, k=5)
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_query = total_time / len(query_points)
        assert avg_time_per_query < 1.0  # Assuming the query time should be less than 1 second

    def test_memory_usage(self, large_data):
        ann = ScannANN()
        ann.build_index(large_data)
        memory_usage = ann.get_index_memory_usage()
        assert memory_usage < 1e9  # Assuming the memory usage should be less than 1GB
