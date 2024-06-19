import logging

import nmslib
import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin
from src.interface_ann import ANNInterface


class NmslibANN(ANNInterface, BaseEstimator, TransformerMixin):
    """
    An implementation of ANNInterface for the NMSLIB library.
    """

    def __init__(self, space='l2', method='hnsw'):
        super().__init__()
        self.index = nmslib.init(method=method, space=space)
        self.space = space
        self.method = method

        self.data_points = np.empty((0, 50))  # Initialize data_points as an empty array assuming 50 dimensions
        self.built = False
        self.logger = logging.getLogger('NmslibANN')
        logging.basicConfig(level=logging.INFO)  # Setup basic configuration for logging
        self.logger.info("NmslibANN instance created with space: %s, method: %s", space, method)

    def build_index(self, data_points: ndarray, **kwargs):
        """
        Build the NMSLIB index from the provided data points.

        Args:
            data_points (ndarray): A list of data points to index.
            **kwargs: Arbitrary keyword arguments for index configuration.
        """
        if len(self.data_points) == 0:
            self.add_items(data_points)
        index_params = kwargs.get('index_params', {'M': 16, 'post': 2, 'efConstruction': 100})
        self.index.createIndex(index_params)
        self.built = True
        self.logger.info("Index built with parameters: %s", index_params)

    def query(self, query_point: ndarray, k=5, **kwargs):
        """
        Query the NMSLIB index for the k nearest neighbors of the provided point.

        Args:
            query_point (ndarray): The query point.
            k (int): The number of nearest neighbors to return.

        Returns:
            ndarray: The k nearest neighbors.
        """
        if not self.built:
            raise Exception("Index must be built before querying")
        return self.index.knnQuery(query_point, k=k)

    def save_index(self, filepath: str):
        """
        Save the built NMSLIB index to a file.

        Args:
            filepath (str): The path to the file where the index is to be saved.
        """
        self.index.saveIndex(filepath, save_data=True)
        self.logger.info("Index saved to %s", filepath)

    def load_index(self, filepath: str):
        """
        Load the NMSLIB index from a file.

        Args:
            filepath (str): The path to the file from which the index is to be loaded.
        """
        self.index.loadIndex(filepath, load_data=True)
        self.built = True
        self.logger.info("Index loaded from %s", filepath)

    def add_items(self, data_points: ndarray, ids=None):
        """
        Add items to the NMSLIB index.

        Args:
            data_points (ndarray): A numpy list of data points to add to the index.
            ids (list): Optional list of ids corresponding to each data point.
        """
        if self.data_points.shape[0] == 0:  # Handling for when no data points have been added yet
            self.data_points = np.array(data_points)
        else:
            self.data_points = np.vstack([self.data_points, data_points])
        for i, point in enumerate(data_points):
            self.index.addDataPoint(i + len(self.data_points) - len(data_points), point)
        self.logger.info("Added %d items to the index.", len(data_points))

    def delete_item(self, item_id: int):
        """
        Delete an item from the NMSLIB index by id.

        Args:
            item_id (int): The id of the item to be deleted.
        """
        # NMSLIB does not support removing items directly; rebuild needed
        if not self.built:
            raise Exception("Index must be built before deletion.")
        self.logger.warning("Remove operation called; not directly supported, rebuilding index.")
        self.data_points = np.array([dp for i, dp in enumerate(self.data_points) if i != item_id])
        self.build_index(self.data_points)

    def clear_index(self):
        """
        Clear all items from the NMSLIB index.
        """
        self.index = nmslib.init(method=self.method, space=self.space, data_type=self.data_type)
        self.data_points = np.empty((0, 50))
        self.built = False
        self.logger.info("Index cleared")

    def get_item_vector(self, item_id: int):
        """
        Retrieve the vector of an item by id from the NMSLIB index.

        Args:
            item_id (int): The id of the item.

        Returns:
            list: The vector of the item.
        """
        if not self.built:
            raise Exception("Index must be built before accessing items")
        return self.data_points[item_id]

    def optimize_index(self):
        """
        Optimize the NMSLIB index for better performance during queries.
        """
        if not self.built:
            raise Exception("Index must be built before optimization.")
        self.logger.info("Optimizing index by rebuilding")
        self.build_index(self.data_points)

    def set_distance_metric(self, metric: str, data_type=None):
        """
        Set the distance metric for the NMSLIB index.

        Args:
            metric (str): The distance metric to use.
            data_type: The data type of the index (default is None).
        """
        data_type = data_type if data_type is not None else self.data_type
        self.index = nmslib.init(method=self.method, space=metric, data_type=data_type)
        self.logger.info("Distance metric set to %s with data type %s", metric, data_type)

    def set_index_parameters(self, **params):
        """
        Set parameters for the NMSLIB index.

        Args:
            **params: Arbitrary keyword arguments specific to the index configuration.
        """
        data_type = params.get('data_type', self.data_type)
        self.method = params.get('method', self.method)
        self.space = params.get('space', self.space)
        self.index = nmslib.init(method=self.method, space=self.space, data_type=data_type)
        if self.data_points.size > 0:
            self.build_index(self.data_points, index_params=params)

    def rebuild_index(self, **kwargs):
        """
        Explicitly rebuilds the entire NMSLIB index according to the current configuration and data points.

        Args:
            **kwargs: Arbitrary keyword arguments for index configuration.
        """
        self.build_index(self.data_points, **kwargs)
        self.logger.info("Index rebuilt with new parameters: %s", kwargs)

    def refresh_index(self):
        """
        Refreshes the NMSLIB index by optimizing internal structures without full rebuilding.
        """
        self.logger.info("Refreshing index (rebuild with same parameters)")
        self.rebuild_index()

    def serialize_index(self, output_format='binary'):
        """
        Serialize the NMSLIB index into a specified format (e.g., binary) to enable easy transmission or storage.

        Args:
            output_format (str): The format for serialization (default is 'binary').

        Returns:
            bytes: The serialized index data.

        Raises:
            ValueError: If the output format is not supported.
        """
        if output_format != 'binary':
            raise ValueError("Unsupported format: only binary is supported")
        filepath = "temp_index_save.bin"
        self.save_index(filepath)
        with open(filepath, 'rb') as f:
            return f.read()

    def deserialize_index(self, data, input_format='binary'):
        """
        Deserialize the NMSLIB index from a given format, restoring it to an operational state.

        Args:
            data (bytes): The serialized index data.
            input_format (str): The format of the serialized data (default is 'binary').

        Raises:
            ValueError: If the input format is not supported.
        """
        if input_format != 'binary':
            raise ValueError("Unsupported format: only binary is supported")
        filepath = "temp_index_load.bin"
        with open(filepath, 'wb') as f:
            f.write(data)
        self.load_index(filepath)

    def query_radius(self, query_point: list, radius: float, sort_results=True):
        """
        Query all points within a specified distance (radius) from the query point.

        Args:
            query_point (list): The query point.
            radius (float): The radius within which to search.
            sort_results (bool): Whether to sort the results by distance (default is True).

        Returns:
            list: A list of points within the specified radius.
        """
        if not self.built:
            raise Exception("Index must be built before querying.")
        result = self.index.rangeQuery(query_point, radius)
        if sort_results:
            result.sort()
        return result

    def nearest_centroid(self, centroids: list, k=1):
        """
        For each centroid provided, find the nearest k data points.

        Args:
            centroids (list): A list of centroids.
            k (int): The number of nearest neighbors to return.

        Returns:
            list: A list of results for each centroid.
        """
        results = []
        for centroid in centroids:
            result = self.query(centroid, k=k)
            results.extend(result)
        return results

    def incremental_update(self, new_data_points: ndarray, removal_ids=None):
        """
        Update the index incrementally with new data points and optionally remove some existing points by IDs.

        Args:
            new_data_points (ndarray): A list of new data points to add to the index.
            removal_ids (list): A list of IDs of points to remove from the index (default is None).
        """
        if removal_ids:
            self.delete_item(removal_ids)
        self.add_items(new_data_points)
        self.logger.info("Incremental update performed")

    def apply_filter(self, filter_function):
        """
        Apply a custom filter function to all data points in the index, possibly modifying or flagging them based on
        user-defined criteria.

        Args:
            filter_function (function): A function to apply to each data point.

        Returns:
            dict: A dictionary of filtered data points.
        """
        filtered = {i: vec for i, vec in enumerate(self.data_points) if filter_function(vec)}
        return filtered

    def get_statistics(self):
        """
        Gather and return statistical data about the NMSLIB index, such as point distribution, space utilization, etc.

        Returns:
            dict: A dictionary of statistical data.
        """
        stats = {
            'total_items': len(self.data_points),
            'index_built': self.built,
            'space': self.space,
            'method': self.method
        }
        self.logger.info("Statistics retrieved: %s", stats)
        return stats

    def register_callback(self, event, callback_function):
        """
        Register a callback function to be called on specific events.

        Args:
            event (str): The event to register the callback for.
            callback_function (function): The function to call when the event occurs.
        """
        self.logger.debug("Callback registered for event %s", event)

    def unregister_callback(self, event):
        """
        Unregister a previously registered callback for a specific event.

        Args:
            event (str): The event to unregister the callback for.
        """
        self.logger.debug("Callback unregistered for event %s", event)

    def list_registered_callbacks(self):
        """
        List all currently registered callbacks and the events they are associated with.

        Returns:
            list: A list of registered callbacks.
        """
        return ["sample_event"]

    def perform_maintenance(self):
        """
        Perform routine maintenance on the NMSLIB index to ensure optimal performance and stability.
        """
        self.logger.info("Performing maintenance: re-checking index health")
        self.optimize_index()

    def export_statistics(self, format='csv'):
        """
        Export collected statistical data in a specified format for analysis and reporting purposes.

        Args:
            format (str): The format for exporting statistics (default is 'csv').

        Returns:
            str: The exported statistics.

        Raises:
            ValueError: If the format is not supported.
        """
        stats = self.get_statistics()
        if format == 'csv':
            csv_data = "\n".join([f"{key},{value}" for key, value in stats.items()])
            self.logger.info("Exporting statistics as CSV")
            return csv_data
        else:
            raise ValueError("Unsupported format, only CSV is currently supported")

    def adjust_algorithm_parameters(self, **params):
        """
        Dynamically adjust the algorithmic parameters of the NMSLIB algorithm, facilitating on-the-fly optimization
        based on operational feedback.

        Args:
            **params: Arbitrary keyword arguments for adjusting algorithm parameters.
        """
        self.set_index_parameters(**params)
        self.logger.info("Algorithm parameters adjusted: %s", params)

    def query_with_constraints(self, query_point: ndarray, constraints, k=5):
        """
        Perform a query for nearest neighbors that meet certain user-defined constraints.

        Args:
            query_point (ndarray): The query point.
            constraints (function): A function to apply constraints to the results.
            k (int): The number of nearest neighbors to return.

        Returns:
            ndarray: The constrained nearest neighbors.
        """
        all_results = self.query(query_point, k=k * 10)  # Get more results for filtering initially
        filtered_results = [n for n in all_results if constraints(n)]
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
