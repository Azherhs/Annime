import os
import logging
import nmslib
import numpy as np

from numpy import ndarray

# from concurrent.futures import ThreadPoolExecutor
from src.interface_ann import ANNInterface
import shutil


class NmslibANN(ANNInterface):
    data_points: ndarray

    def __init__(self, space='l2', method='hnsw', dtype=nmslib.DistType.FLOAT):
        super().__init__()
        self.space = space
        self.method = method
        self.dtype = dtype
        self.index = nmslib.init(method=method, space=space, dtype=dtype)
        self.data_points = np.empty((0, 50))  # Initialize data_points as an empty array assuming 50 dimensions
        self.built = False
        # Initialize logger here to ensure it is available in all methods
        self.logger = logging.getLogger('NmslibANN')
        logging.basicConfig(level=logging.INFO)  # Setup basic configuration for logging
        self.logger.info("NmslibANN instance created with space: %s, method: %s", space, method)


    def build_index(self, data_points, **kwargs):
        if len(self.data_points) == 0:
            self.add_items(data_points)
        index_params = kwargs.get('index_params', {'M': 16, 'post': 2, 'efConstruction': 100})
        self.index.createIndex(index_params)
        self.built = True
        self.logger.info("Index built with parameters: %s", index_params)

    def query(self, query_point, k=5, **kwargs):
        if not self.built:
            raise Exception("Index must be built before querying")
        return self.index.knnQuery(query_point, k=k)

    def save_index(self, filepath):
        self.index.saveIndex(filepath, save_data=True)
        self.logger.info("Index saved to %s", filepath)

    def load_index(self, filepath):
        self.index.loadIndex(filepath, load_data=True)
        self.built = True
        self.logger.info("Index loaded from %s", filepath)

    def add_items(self, data_points):
        if self.data_points.shape[0] == 0:  # Handling for when no data points have been added yet
            self.data_points = np.array(data_points)
        else:
            self.data_points = np.vstack([self.data_points, data_points])
        for i, point in enumerate(data_points):
            self.index.addDataPoint(i + len(self.data_points) - len(data_points), point)



    def remove_items(self, ids):
        # Ensure ids is a list if not convert numpy array to list
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        # NMSLIB does not support removing items directly from an index; rebuild needed
        self.logger.warning("Remove operation called; not directly supported, rebuilding index")
        self.data_points = [dp for i, dp in enumerate(self.data_points) if i not in ids]
        self.build_index(self.data_points)

    def update_item(self, item_id, new_vector):
        # Direct item update not supported; simulate by rebuilding the index
        self.logger.warning("Update operation called; not directly supported, rebuilding index")
        self.data_points[item_id] = new_vector
        self.build_index(self.data_points)

    def get_item_vector(self, item_id):
        if not self.built:
            raise Exception("Index must be built before accessing items")
        return self.data_points[item_id]

    def optimize_index(self):
        # NMSLIB optimizes during index creation; simulate re-optimization by rebuilding
        self.logger.info("Optimizing index by rebuilding")
        self.build_index(self.data_points)

    def set_distance_metric(self, metric, dtype=None):
        self.space = metric
        dtype = dtype if dtype is not None else self.dtype
        self.index = nmslib.init(method=self.method, space=metric, dtype=dtype)
        self.logger.info("Distance metric set to %s with dtype %s", metric, dtype)

    def set_index_parameters(self, **params):
        dtype = params.get('dtype', self.dtype)  # Use dtype from params or fallback to the instance attribute
        self.method = params.get('method', self.method)
        self.space = params.get('space', self.space)
        self.index = nmslib.init(method=self.method, space=self.space, dtype=dtype)
        if self.data_points.size > 0:
            self.build_index(self.data_points, index_params=params)

    def clear_index(self):
        dtype = self.dtype
        self.index = nmslib.init(method=self.method, space=self.space, dtype=dtype)
        self.data_points = []
        self.built = False
        self.logger.info("Index cleared")

    def rebuild_index(self, **kwargs):
        self.build_index(self.data_points, **kwargs)
        self.logger.info("Index rebuilt with new parameters: %s", kwargs)

    def refresh_index(self):
        self.logger.info("Refreshing index (rebuild with same parameters)")
        self.rebuild_index()

    def serialize_index(self, output_format='binary'):
        if output_format != 'binary':
            raise ValueError("Unsupported format: only binary is supported")
        filepath = "temp_index_save.bin"
        self.save_index(filepath)
        with open(filepath, 'rb') as f:
            return f.read()

    def deserialize_index(self, data, input_format='binary'):
        if input_format != 'binary':
            raise ValueError("Unsupported format: only binary is supported")
        filepath = "temp_index_load.bin"
        with open(filepath, 'wb') as f:
            f.write(data)
        self.load_index(filepath)

    def query_radius(self, query_point, radius, sort_results=True):
        # Example using NMSLIB's rangeQuery feature
        if not self.built:
            raise Exception("Index must be built before querying.")
        result = self.index.rangeQuery(query_point, radius)
        if sort_results:
            result.sort()
        return result

    def nearest_centroid(self, centroids, k=1):
        results = []
        for centroid in centroids:
            result = self.query(centroid, k=k)
            results.extend(result)
        return results

    def incremental_update(self, new_data_points, removal_ids=None):
        if removal_ids:
            self.remove_items(removal_ids)
        self.add_items(new_data_points)
        self.logger.info("Incremental update performed")

    def apply_filter(self, filter_function):
        filtered = {i: vec for i, vec in enumerate(self.data_points) if filter_function(vec)}
        return filtered

    def get_statistics(self):
        stats = {
            'total_items': len(self.data_points),
            'index_built': self.built,
            'space': self.space,
            'method': self.method
        }
        self.logger.info("Statistics retrieved: %s", stats)
        return stats

    def register_callback(self, event, callback_function):
        # Placeholder for actual callback mechanism implementation
        self.logger.debug("Callback registered for event %s", event)

    def unregister_callback(self, event):
        # Placeholder for actual callback mechanism removal
        self.logger.debug("Callback unregistered for event %s", event)

    def list_registered_callbacks(self):
        # Placeholder for listing current callbacks
        return ["sample_event"]

    def perform_maintenance(self):
        # Simulate maintenance by checking and optionally rebuilding the index
        self.logger.info("Performing maintenance: re-checking index health")
        self.optimize_index()

    def export_statistics(self, output_format='csv'):
        stats = self.get_statistics()
        if output_format == 'csv':
            csv_data = "\n".join([f"{key},{value}" for key, value in stats.items()])
            self.logger.info("Exporting statistics as CSV")
            return csv_data
        else:
            raise ValueError("Unsupported format, only CSV is currently supported")

    def adjust_algorithm_parameters(self, **params):
        self.set_index_parameters(**params)
        self.logger.info("Algorithm parameters adjusted: %s", params)

    def query_with_constraints(self, query_point, constraints, k=5):
        all_results = self.query(query_point, k=k * 10)  # Get more results for filtering
        filtered_results = [res for res in all_results if constraints(res)]
        return filtered_results[:k]

    def batch_query(self, query_points, k=5, include_distances=False):
        results = []
        for query_point in query_points:
            result = self.query(query_point, k=k)
            if include_distances:
                results.append(result)
            else:
                results.append([x[0] for x in result])
        return results

    def parallel_query(self, query_points, k=5, num_threads=4):
        # Implement using Python's multiprocessing for true parallelism
        from multiprocessing import Pool
        with Pool(processes=num_threads) as pool:
            results = pool.starmap(self.query, [(point, k) for point in query_points])
        return results

    def benchmark_performance(self, queries, k=5, rounds=10):
        # Detailed performance benchmarking using time measurement
        import time
        start_time = time.time()
        for _ in range(rounds):
            for query in queries:
                self.query(query, k=k)
        end_time = time.time()
        duration = (end_time - start_time) / (len(queries) * rounds)
        self.logger.info("Benchmark completed: Avg query time = %.5f seconds", duration)
        return f"Average query time: {duration:.5f} seconds"

    def export_to_dot(self, filepath):
        # Simulate DOT export if not natively supported
        self.logger.warning("DOT export not supported, using placeholder")
        with open(filepath, 'w') as f:
            f.write("digraph G {\n")
            for i in range(len(self.data_points)):
                f.write(f"    {i} [label=\"{i}\"];\n")
            f.write("}\n")
        self.logger.info("Index structure exported to DOT file at %s", filepath)

    def enable_logging(self, level='INFO'):
        # Setup logging configuration
        logging.basicConfig(level=getattr(logging, level.upper()))
        self.logger.setLevel(level.upper())
        self.logger.info("Logging enabled at level: %s", level)

    def delete_item(self, item_id):
        # NMSLIB does not support deleting items directly; need to rebuild the index
        if item_id < 0 or item_id >= len(self.data_points):
            raise ValueError("Invalid item_id")
        self.data_points = [dp for i, dp in enumerate(self.data_points) if i != item_id]
        self.build_index(self.data_points)  # Rebuild the index without the specified item

    def get_index_size(self):
        return len(self.data_points)

    def get_index_memory_usage(self):
        """ Estimate memory usage by saving the index to a temporary file and checking its size. """
        temp_path = 'temp_nmslib_index.bin'
        self.save_index(temp_path)
        memory_usage = os.path.getsize(temp_path)
        os.remove(temp_path)
        return memory_usage

    def backup_index(self, backup_location):
        """
        Backup the current state of the index to a specified location.
        This involves copying the index data file to a backup location.
        """
        if not self.built:
            raise Exception("Index must be built before it can be backed up.")
        original_path = 'current_nmslib_index.bin'
        self.save_index(original_path)
        shutil.copy(original_path, backup_location)
        os.remove(original_path)  # Cleanup the temporary file
        print(f"Index backed up to {backup_location}")

    def restore_index_from_backup(self, backup_location):
        """
        Restore the index state from a backup file.
        This method loads the index from the specified backup file.
        """
        if not os.path.exists(backup_location):
            raise FileNotFoundError(f"No backup found at {backup_location}")
        self.load_index(backup_location)
        print(f"Index restored from {backup_location}")

    # Example of creating and using the NmslibANN class
    def example_usage(self):
        data = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        self.build_index(data)
        print("Index Size:", self.get_index_size())
        print("Memory Usage:", self.get_index_memory_usage())
        self.backup_index('nmslib_backup.bin')
        self.restore_index_from_backup('nmslib_backup.bin')
        results = self.batch_query([[0.2, 0.2]], k=2)
        print("Batch Query Results:", results)


if __name__ == "__main__":
    nmslib_ann = NmslibANN()
    data = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    nmslib_ann.add_items(data)
    nmslib_ann.build_index(data)
    nmslib_ann.save_index('complex_nmslib_index.nms')
    nmslib_ann.load_index('complex_nmslib_index.nms')
    print("Done processing")