import pytest
import numpy as np
import os
import sys
from test_utils import verify_build_and_query, verify_save_and_load

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../annime')))
from annime.annoy_int import AnnoyANN


@pytest.fixture(scope='module')
def data():
    return np.random.rand(100, 10).astype(np.float32)


class TestAnnoyANN:
    @pytest.fixture(autouse=True)
    def setup_annoy(self, data):
        self.ann = AnnoyANN(dim=10, num_trees=10)
        self.data = data

    def test_build_and_query(self):
        # Update the test to verify that 5 nearest neighbors are returned
        k = 5  # Number of nearest neighbors to query
        ann = self.ann
        data = self.data

        ann.build_index(data)
        assert ann.built
        assert ann.get_index_size() == len(data)

        # Verify the query returns the correct number of neighbors
        query_point = data[0]
        result, _ = ann.query(query_point, k=k)
        assert len(result) == k, f"Expected {k} neighbors, but got {len(result)}"
