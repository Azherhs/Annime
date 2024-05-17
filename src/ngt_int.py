import ngtpy
import numpy as np
from src.interface_ann import ANNInterface
import logging
import time


class NgtANN(ANNInterface):
    """
    An implementation of ANNInterface for the NGT (Neighborhood Graph and Tree) library.
    """

    def __init__(self, dim, distance_type="L2"):
        super().__init__()
        self.index = ngtpy.Index(path=".", dimension=dim, distance_type=distance_type)
        self.dim = dim
        self.distance_type = distance_type
        self.built = False
        self.logger = logging.getLogger(__name__)

    def build_index(self, data_points, **kwargs):
        for i, vector in enumerate(data_points):
            self.index.insert(vector)
        self.index.build_index()
        self.built = True
        self.logger.info("Index built with %d data points.", len(data_points))

    def query(self, query_point, k=5, **kwargs):
        if not self.built:
            raise Exception("Index must be built before querying.")
        return self.index.search(query_point, size=k, with_distance=True)

    def save_index(self, filepath):
        if not self.built:
            raise Exception("Index must be built before it can be saved.")
        self.index.save(filepath)
        self.logger.info("Index saved to %s.", filepath)

    def load_index(self, filepath):
        self.index = ngtpy.Index(filepath)
        self.built = True
        self.logger.info("Index loaded from %s.", filepath)

    def set_distance_metric(self, metric):
        self.distance_type = metric
        self.index = ngtpy.Index(path=".", dimension=self.dim, distance_type=metric)
        self.logger.info("Distance metric set to %s.", metric)

    def set_index_parameters(self, **params):
        for param, value in params.items():
            setattr(self.index.property, param, value)
        self.logger.info("Index parameters set: %s", params)

    def add_items(self, data_points, ids=None):
        if ids and len(data_points) != len(ids):
            raise ValueError("Length of data_points and ids must match.")
        for i, vector in enumerate(data_points):
            idx = ids[i] if ids else i
            self.index.insert(vector, object_id=idx)
        self.logger.info("Added %d items to the index.", len(data_points))

    def delete_item(self, item_id):
        self.index.remove(item_id)
        self.logger.info("Item with id %d deleted from the index.", item_id)

    def clear_index(self):
        self.index.clear()
        self.built = False
        self.logger.info("Index cleared.")

    def get_item_vector(self, item_id):
        return self.index.get_object(item_id)

    def optimize_index(self):
        self.index.build_index()
        self.logger.info("Index optimized.")

    def get_index_size(self):
        return self.index.size()

    def get_index_memory_usage(self):
        return self.index.memory()

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
        from multiprocessing import Pool
        with Pool(processes=num_threads) as pool:
            results = pool.starmap(self.query, [(query_point, k) for query_point in query_points])
        return results

    def benchmark_performance(self, queries, k=5, rounds=10):

        start_time = time.time()
        for _ in range(rounds):
            for query in queries:
                self.query(query, k=k)
        end_time = time.time()
        duration = (end_time - start_time) / (len(queries) * rounds)
        self.logger.info("Benchmark completed: Avg query time = %.5f seconds", duration)
        return f"Average query time: {duration:.5f} seconds"

    def export_to_dot(self, filepath):
        raise NotImplementedError("NGT does not support exporting to DOT format.")

    def enable_logging(self, level='INFO'):
        logging.basicConfig(level=getattr(logging, level.upper()))
        self.logger.setLevel(level.upper())
        self.logger.info("Logging enabled at level: %s", level)

    def rebuild_index(self, **kwargs):
        self.index.build_index()
        self.logger.info("Index rebuilt.")

    def refresh_index(self):
        self.index.build_index()
        self.logger.info("Index refreshed.")

    def serialize_index(self, output_format='binary'):
        if output_format != 'binary':
            raise ValueError("NGT currently supports only binary serialization format.")
        temp_path = "temp_ngt_index"
        self.save_index(temp_path)
        with open(temp_path, 'rb') as file:
            serialized_data = file.read()
        return serialized_data

    def deserialize_index(self, data, input_format='binary'):
        if input_format != 'binary':
            raise ValueError("NGT currently supports only binary deserialization format.")
        temp_path = "temp_load_ngt_index"
        with open(temp_path, 'wb') as file:
            file.write(data)
        self.load_index(temp_path)

    def query_radius(self, query_point, radius, sort_results=True):
        if not self.built:
            raise Exception("Index must be built before querying.")
        result = self.index.search(query_point, size=10000, radius=radius)
        if sort_results:
            result = sorted(result, key=lambda x: x[1])
        return result

    def nearest_centroid(self, centroids, k=1):
        results = []
        for centroid in centroids:
            neighbors = self.index.search(centroid, size=k, with_distance=True)
            results.append(neighbors)
        return results

    def incremental_update(self, new_data_points, removal_ids=None):
        if removal_ids:
            for item_id in removal_ids:
                self.delete_item(item_id)
        self.add_items(new_data_points)
        self.logger.info("Incremental update performed.")

    def backup_index(self, backup_location):
        if not self.built:
            raise Exception("Index must be built before it can be backed up.")
        self.save_index(backup_location)
        self.logger.info("Index backed up to %s.", backup_location)

    def restore_index_from_backup(self, backup_location):
        self.load_index(backup_location)
        self.built = True
        self.logger.info("Index restored from %s.", backup_location)

    def apply_filter(self, filter_function):
        filtered = {i: vec for i, vec in enumerate(self.index.get_objects()) if filter_function(vec)}
        return filtered

    def get_statistics(self):
        stats = {
            'num_items': self.index.size(),
            'index_built': self.built,
            'distance_type': self.distance_type
        }
        self.logger.info("Statistics retrieved: %s", stats)
        return stats

    def register_callback(self, event, callback_function):
        pass

    def unregister_callback(self, event):
        pass

    def list_registered_callbacks(self):
        return []

    def perform_maintenance(self):
        self.index.build_index()
        self.logger.info("Maintenance performed: index verified.")

    def export_statistics(self, format='csv'):
        stats = self.get_statistics()
        if format == 'csv':
            csv_data = "\n".join([f"{key},{value}" for key, value in stats.items()])
            return csv_data
        else:
            raise ValueError("Unsupported format")

    def adjust_algorithm_parameters(self, **params):
        for param, value in params.items():
            setattr(self.index.property, param, value)
        self.logger.info("Algorithm parameters adjusted: %s", params)

    def query_with_constraints(self, query_point, constraints, k=5):
        all_results = self.query(query_point, k=k * 10)
        filtered_results = [res for res in all_results if constraints(res)]
        return filtered_results[:k]
