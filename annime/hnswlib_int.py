import hnswlib
import numpy as np
from annime.interface_ann import ANNInterface
from sklearn.base import BaseEstimator, TransformerMixin
import logging
import os
import pickle
import time


class HnswlibANN(ANNInterface, BaseEstimator, TransformerMixin):
    """
    An implementation of ANNInterface for the HNSWLIB library.
    """

    def __init__(self, dim, space='l2', max_elements=10000):
        super().__init__()
        self.dim = dim
        self.space = space
        self.max_elements = max_elements
        self.index = hnswlib.Index(space=space, dim=dim)
        self.data_points = None
        self.built = False
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.logger.info("HnswlibANN instance created with space: %s", space)

    def build_index(self, data_points: np.ndarray, **kwargs):
        """
        Build the HNSWLIB index from the provided data points.

        Args:
            data_points (np.ndarray): A list of data points to index.
            **kwargs: Arbitrary keyword arguments for index configuration.
        """
        self.data_points = data_points
        num_elements = data_points.shape[0]
        self.index.init_index(max_elements=num_elements, **kwargs)
        self.index.add_items(data_points, np.arange(num_elements))
        self.built = True
        self.logger.info("Index built with %d data points.", num_elements)

    def query(self, query_point: np.ndarray, k=5, **kwargs):
        """
        Query the HNSWLIB index for the k nearest neighbors of the provided point.

        Args:
            query_point (np.ndarray): The query point.
            k (int): The number of nearest neighbors to return.

        Returns:
            tuple: A tuple containing the k nearest neighbors and their distances.
        """
        if not self.built:
            raise ValueError("Index not built. Call build_index first.")
        labels, distances = self.index.knn_query(query_point, k=k, filter=None)
        return labels[0].tolist(), distances[0].tolist()

    def save_index(self, filepath: str):
        """
        Save the built HNSWLIB index to a file.

        Args:
            filepath (str): The path to the file where the index is to be saved.
        """
        if not self.built:
            raise ValueError("Index not built. Call build_index first.")
        self.index.save_index(filepath)
        with open(filepath + '_data.pkl', 'wb') as f:
            pickle.dump(self.data_points, f)
        self.logger.info("Index saved to %s.", filepath)

    def load_index(self, filepath: str):
        """
        Load the HNSWLIB index from a file.

        Args:
            filepath (str): The path to the file from which the index is to be loaded.
        """
        self.index.load_index(filepath)
        with open(filepath + '_data.pkl', 'rb') as f:
            self.data_points = pickle.load(f)
        self.built = True
        self.logger.info("Index loaded from %s.", filepath)

    def set_distance_metric(self, metric: str):
        """
        Set the distance metric for the HNSWLIB index.

        Args:
            metric (str): The distance metric to use (e.g., 'l2', 'ip').
        """
        self.space = metric
        self.logger.info("Distance metric set to %s.", metric)

    def set_index_parameters(self, **params):
        """
        Set parameters for the HNSWLIB index.

        Args:
            **params: Arbitrary keyword arguments specific to the HNSWLIB index configuration.
        """
        self.max_elements = params.get('max_elements', self.max_elements)
        if self.built:
            self.build_index(self.data_points, **params)
        self.logger.info("Index parameters set: %s", params)

    def add_items(self, data_points: np.ndarray, ids=None):
        """
        Add items to the HNSWLIB index.

        Args:
            data_points (np.ndarray): A numpy list of data points to add to the index.
            ids (list): Optional list of ids corresponding to each data point.
        """
        if not self.built:
            raise ValueError("Index not built. Call build_index first.")
        if ids is None:
            ids = np.arange(len(self.data_points), len(self.data_points) + len(data_points))
        self.index.add_items(data_points, ids)
        self.data_points = np.vstack([self.data_points, data_points])
        self.logger.info("Added %d items to the index.", len(data_points))

    def delete_item(self, item_id: int):
        """
        Delete an item from the HNSWLIB index by id.

        Args:
            item_id (int): The id of the item to be deleted.

        Raises:
            NotImplementedError: HNSWLIB does not support item deletion directly.
        """
        raise NotImplementedError("HNSWLIB does not support removing items directly")

    def clear_index(self):
        """
        Clear all items from the HNSWLIB index.
        """
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.data_points = None
        self.built = False
        self.logger.info("Index cleared.")

    def get_item_vector(self, item_id: int):
        """
        Retrieve the vector of an item by id from the HNSWLIB index.

        Args:
            item_id (int): The id of the item.

        Returns:
            np.ndarray: The vector of the item.
        """
        if self.data_points is None or item_id >= len(self.data_points):
            raise ValueError("Invalid item_id or index not built.")
        return self.data_points[item_id]

    def optimize_index(self):
        """
        Optimize the HNSWLIB index for better performance during queries.
        """
        # HNSWLIB indices are optimized on-the-fly; this is a placeholder for more complex indices
        self.logger.info("Index optimized.")

    def get_index_size(self):
        """
        Return the current size of the HNSWLIB index in terms of the number of items.

        Returns:
            int: The number of items in the index.
        """
        return len(self.data_points) if self.data_points is not None else 0

    def get_index_memory_usage(self):
        """
        Return the amount of memory used by the HNSWLIB index.

        Returns:
            int: The memory usage of the index.
        """
        if not self.built:
            raise ValueError("Index not built. Call build_index first.")
        # Estimate memory usage by saving the index to a temporary file and checking its size
        temp_path = 'temp_hnswlib_index.bin'
        self.save_index(temp_path)
        memory_usage = os.path.getsize(temp_path)
        os.remove(temp_path)
        return memory_usage

    def batch_query(self, query_points: np.ndarray, k=5, include_distances=False):
        """
        Perform a batch query for multiple points, returning their k nearest neighbors.

        Args:
            query_points (np.ndarray): A list of query points.
            k (int): The number of nearest neighbors to return.
            include_distances (bool): Whether to include distances in the results.

        Returns:
            list: A list of results for each query point.
        """
        if not self.built:
            raise ValueError("Index not built. Call build_index first.")
        labels, distances = self.index.knn_query(query_points, k=k, filter=None)
        if include_distances:
            return list(zip(labels.tolist(), distances.tolist()))
        else:
            return labels.tolist()

    def benchmark_performance(self, queries: np.ndarray, k=5, rounds=10):
        """
        Benchmark the query performance of the index with a set of queries repeated over several rounds.

        Args:
            queries (np.ndarray): A list of query points.
            k (int): The number of nearest neighbors to return.
            rounds (int): The number of rounds to repeat the benchmark.

        Returns:
            str: The average query time.
        """
        if not self.built:
            raise ValueError("Index not built. Call build_index first.")
        times = []
        for _ in range(rounds):
            start_time = time.time()
            self.index.knn_query(queries, k, filter=None)
            end_time = time.time()
            times.append(end_time - start_time)
        avg_time = sum(times) / (len(queries) * rounds)
        self.logger.info("Benchmark completed: Avg query time = %.5f seconds", avg_time)
        return f"Average query time: {avg_time:.5f} seconds"

    def export_to_dot(self, filepath: str):
        """
        Export the structure of the HNSWLIB index to a DOT file for visualization.

        Args:
            filepath (str): The path to the file where the DOT representation is to be saved.

        Raises:
            NotImplementedError: HNSWLIB does not support exporting to DOT format.
        """
        raise NotImplementedError("Export to DOT format is not supported by HNSWLIB.")

    def enable_logging(self, level='INFO'):
        """
        Enable detailed logging of operations within the HNSWLIB index.

        Args:
            level (str): The logging level (e.g., 'INFO', 'DEBUG').
        """
        logging.basicConfig(level=getattr(logging, level.upper()))
        self.logger.setLevel(level.upper())
        self.logger.info("Logging enabled at level: %s", level)

    def rebuild_index(self, **kwargs):
        """
        Explicitly rebuilds the entire HNSWLIB index according to the current configuration and data points.

        Args:
            **kwargs: Arbitrary keyword arguments for index configuration.
        """
        if not self.built:
            raise ValueError("Index not built. Call build_index first.")
        self.build_index(self.data_points, **kwargs)
        self.logger.info("Index rebuilt with parameters: %s", kwargs)

    def refresh_index(self):
        """
        Refreshes the HNSWLIB index by optimizing internal structures without full rebuilding.

        Note: HNSWLIB does not have a native refresh method; simulate by rebuilding.
        """
        self.rebuild_index()

    def serialize_index(self, output_format='binary'):
        """
        Serialize the HNSWLIB index into a specified format (e.g., binary) to enable easy transmission or storage.

        Args:
            output_format (str): The format for serialization (default is 'binary').

        Returns:
            bytes: The serialized index data.

        Raises:
            ValueError: If the output format is not supported.
        """
        if output_format != 'binary':
            raise ValueError("HNSWLIB currently supports only binary serialization format.")
        temp_path = "temp_hnswlib_index"
        self.save_index(temp_path)
        with open(temp_path, 'rb') as file:
            serialized_data = file.read()
        return serialized_data

    def deserialize_index(self, data: bytes, input_format='binary'):
        """
        Deserialize the HNSWLIB index from a given format, restoring it to an operational state.

        Args:
            data (bytes): The serialized index data.
            input_format (str): The format of the serialized data (default is 'binary').

        Raises:
            ValueError: If the input format is not supported.
        """
        if input_format != 'binary':
            raise ValueError("HNSWLIB currently supports only binary deserialization format.")
        temp_path = "temp_load_hnswlib_index"
        with open(temp_path, 'wb') as file:
            file.write(data)
        self.load_index(temp_path)

    def query_radius(self, query_point: np.ndarray, radius: float, sort_results=True, filter=None):
        """
        Query all points within a specified distance (radius) from the query point.

        Args:
            query_point (np.ndarray): The query point.
            radius (float): The radius within which to search.
            sort_results (bool): Whether to sort the results by distance (default is True).
            filter (function): A function to filter the results.

        Returns:
            list: A list of points within the specified radius.
        """
        raise NotImplementedError("HNSWLIB does not support radius-based querying.")

    def nearest_centroid(self, centroids: list, k=1, filter=None):
        """
        For each centroid provided, find the nearest k data points.

        Args:
            centroids (list): A list of centroids.
            k (int): The number of nearest neighbors to return.
            filter (function): A function to filter the results.

        Returns:
            list: A list of results for each centroid.
        """
        if not self.built:
            raise ValueError("Index not built. Call build_index first.")
        nearest_points = []
        for centroid in centroids:
            neighbors = self.query(centroid, k=k)
            if filter:
                neighbors = [n for n in neighbors if filter(n)]
            nearest_points.append(neighbors)
        return nearest_points

    def incremental_update(self, new_data_points: np.ndarray, removal_ids=None, filter=None):
        """
        Update the index incrementally with new data points and optionally remove some existing points by IDs.

        Args:
            new_data_points (np.ndarray): A list of new data points to add to the index.
            removal_ids (list): A list of IDs of points to remove from the index (default is None).
            filter (function): A function to filter the results.

        Raises:
            NotImplementedError: HNSWLIB does not support incremental updates.
        """
        raise NotImplementedError("HNSWLIB does not support incremental updates.")

    def backup_index(self, backup_location: str, filter=None):
        """
        Create a backup of the current index state to a specified location.

        Args:
            backup_location (str): The path to the backup location.
            filter (function): A function to filter the results.
        """
        if not self.built:
            raise ValueError("Index not built. Call build_index first.")
        self.save_index(os.path.join(backup_location, 'hnswlib_index.bin'))
        with open(os.path.join(backup_location, 'hnswlib_data.pkl'), 'wb') as f:
            pickle.dump(self.data_points, f)
        self.logger.info("Index backed up to %s.", backup_location)

    def restore_index_from_backup(self, backup_location: str, filter=None):
        """
        Restore the index state from a backup located at the specified location.

        Args:
            backup_location (str): The path to the backup location.
            filter (function): A function to filter the results.
        """
        self.load_index(os.path.join(backup_location, 'hnswlib_index.bin'))
        with open(os.path.join(backup_location, 'hnswlib_data.pkl'), 'rb') as f:
            self.data_points = pickle.load(f)
        self.built = True
        self.logger.info("Index restored from %s.", backup_location)

    def apply_filter(self, filter_function):
        """
        Apply a custom filter function to all data points in the index, possibly modifying or flagging them based on
        user-defined criteria.

        Args:
            filter_function (function): A function to apply to each data point.

        Returns:
            dict: A dictionary of filtered data points.
        """
        if not callable(filter_function):
            raise ValueError("The filter_function parameter must be callable.")
        filtered = {i: vec for i, vec in enumerate(self.data_points) if filter_function(vec)}
        return filtered

    def get_statistics(self, filter=None):
        """
        Gather and return statistical data about the HNSWLIB index, such as point distribution, space utilization, etc.

        Args:
            filter (function): A function to filter the results.

        Returns:
            dict: A dictionary of statistical data.
        """
        if not self.built:
            raise ValueError("Index not built. Call build_index first.")
        stats = {
            'num_points': len(self.data_points),
            'index_built': self.built,
            'space': self.space,
        }
        self.logger.info("Statistics retrieved: %s", stats)
        return stats

    def perform_maintenance(self, filter=None):
        """
        Perform routine maintenance on the HNSWLIB index to ensure optimal performance and stability.

        Args:
            filter (function): A function to filter the results.
        """
        self.logger.info("Performing maintenance: re-checking index health.")
        self.refresh_index()

    def export_statistics(self, format='csv', filter=None):
        """
        Export collected statistical data in a specified format for analysis and reporting purposes.

        Args:
            format (str): The format for exporting statistics (default is 'csv').
            filter (function): A function to filter the results.

        Returns:
            str: The exported statistics.

        Raises:
            ValueError: If the format is not supported.
        """
        stats = self.get_statistics(filter=filter)
        if format == 'csv':
            csv_data = "\n".join([f"{key},{value}" for key, value in stats.items()])
            self.logger.info("Exporting statistics as CSV.")
            return csv_data
        else:
            raise ValueError("Unsupported format. Only 'csv' is currently supported.")

    def adjust_algorithm_parameters(self, filter=None, **params):
        """
        Dynamically adjust the algorithmic parameters of the underlying HNSWLIB algorithm,
        facilitating on-the-fly optimization based on operational feedback.

        Args:
            filter (function): A function to filter the results.
            **params: Arbitrary keyword arguments for adjusting algorithm parameters.
        """
        self.set_index_parameters(**params)
        self.logger.info("Algorithm parameters adjusted: %s", params)

    def query_with_constraints(self, query_point: np.ndarray, constraints, k=5, filter=None):
        """
        Perform a query for nearest neighbors that meet certain user-defined constraints.

        Args:
            query_point (np.ndarray): The query point.
            constraints (function): A function to apply constraints to the results.
            k (int): The number of nearest neighbors to return.
            filter (function): A function to filter the results.

        Returns:
            list: The constrained nearest neighbors.
        """
        if not callable(constraints):
            raise ValueError("The constraints parameter must be callable.")
        all_results, all_distances = self.query(query_point, k=k * 10)  # Get more results for filtering
        filtered_results = [(n, d) for n, d in zip(all_results, all_distances) if constraints(n)]
        return filtered_results[:k]  # Return only k results after filtering

    def fit(self, X, y=None, **kwargs):
        """
        Fit the Annoy index with the provided data.

        Args:
            X (ndarray): Training data.
            y (ndarray): Training labels (optional).
            **kwargs: Additional parameters for building the index.

        Returns:
            self
        """
        self.build_index(X, **kwargs)
        return self

    def transform(self, X, k=1, **kwargs):
        """
        Transform the data using the Annoy index by querying the nearest neighbors.

        Args:
            X (ndarray): Data to transform.
            k (int): Number of nearest neighbors to query.
            **kwargs: Additional parameters for querying the index.

        Returns:
            ndarray: Indices of the nearest neighbors.
        """
        results = np.array([self.query(x, k=k, **kwargs)[0] for x in X], dtype=int)
        return results

    def fit_transform(self, X, y=None, **kwargs):
        """
        Fit the Annoy index with the provided data and transform it.

        Args:
            X (ndarray): Training data.
            y (ndarray): Training labels (optional).
            **kwargs: Additional parameters for building and querying the index.

        Returns:
            ndarray: Indices of the nearest neighbors.
        """
        return self.fit(X, y, **kwargs).transform(X, **kwargs)
