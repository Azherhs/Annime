import pytest
import numpy as np
import os
import time
import gc


def verify_build_and_query(ann, data, k=5):
    """
    Verify that the ANN instance can build an index and query it correctly.

    Args:
        ann: The ANN instance.
        data (np.ndarray): The data points to index.
        k (int): Number of nearest neighbors to query.
    """
    ann.build_index(data)
    assert ann.built
    assert ann.get_index_size() == len(data)
    query_point = data[0]
    result, _ = ann.query(query_point, k=k)
    assert len(result) == k, f"Expected {k} neighbors, but got {len(result)}"


def verify_save_and_load(ann, data, filename):
    """
    Verify that the ANN instance can save and load an index.

    Args:
        ann: The ANN instance.
        data (np.ndarray): The data points to index.
        filename (str): The filename for saving and loading the index.
    """
    ann.build_index(data)
    ann.save_index(filename)
    ann.load_index(filename)
    del ann.index  # Explicitly delete the index object to release file handle
    gc.collect()  # Force garbage collection to release file handle
    time.sleep(1)  # Wait a bit for the file system to release the handle
    os.remove(filename)  # Clean up the saved file


@pytest.fixture(scope='module')
def data():
    return np.random.rand(100, 10)


@pytest.fixture(scope='module')
def large_data():
    return np.random.rand(10000, 10)


@pytest.fixture(scope='module')
def query_points():
    return np.random.rand(10, 10)
