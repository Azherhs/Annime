import pytest
import numpy as np


def verify_build_and_query(ann, data):
    """
    Verify that the ANN instance can build an index and query it correctly.

    Args:
        ann: The ANN instance.
        data (np.ndarray): The data points to index.
    """
    ann.build_index(data)
    assert ann.built
    assert ann.get_index_size() == len(data)
    query_point = data[0]
    result = ann.query(query_point, k=5)
    assert len(result) == 5


def verify_save_and_load(ann, data, index_name):
    """
    Verify that the ANN instance can save and load an index correctly.

    Args:
        ann: The ANN instance.
        data (np.ndarray): The data points to index.
        index_name (str): The name of the file to save the index.
    """
    ann.build_index(data)
    ann.save_index(index_name)
    ann.load_index(index_name)
    assert ann.built


@pytest.fixture(scope='module')
def data():
    return np.random.rand(100, 10)


@pytest.fixture(scope='module')
def large_data():
    return np.random.rand(10000, 10)


@pytest.fixture(scope='module')
def query_points():
    return np.random.rand(10, 10)
