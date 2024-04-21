from src.nmslib_int import NmslibANN
import numpy as np
import logging
import nmslib

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("NmslibComplexExample")


def create_and_configure_nmslib_ann(space='cosinesimil', method='hnsw', dtype=nmslib.DistType.FLOAT):
    # Initialize the NMSLIB interface with correct data type
    ann = NmslibANN(space=space, method=method, dtype=dtype)
    logger.info(f"Initialized NMSLIB ANN with space {space}, method {method}, and data type {dtype}.")
    return ann


# Generate random high-dimensional data points
np.random.seed(42)
data_points = np.random.rand(1000, 40).astype(np.float32)  # 1000 points in 40 dimensions
query_points = np.random.rand(10, 40).astype(np.float32)  # 10 query points

# Initialize NMSLIB interface
nmslib_ann = create_and_configure_nmslib_ann()

# Add data to the index
nmslib_ann.add_items(data_points)
logger.info("Data points added to the index.")

# Build the index with specific parameters
nmslib_ann.build_index(data_points, index_params={'M': 30, 'post': 0, 'efConstruction': 100})
logger.info("Index built with custom parameters.")

# Perform a batch of complex queries
complex_results = nmslib_ann.batch_query(query_points, k=5)
logger.info("Performed batch querying.")

# Adjust the method and rebuild the index
# nmslib_ann.set_index_parameters(method='sw-graph', space='l2')
# nmslib_ann.build_index(data_points)
# logger.info("Index parameters adjusted and index rebuilt.")

# Simulate an update by adding new data points and rebuilding the index
new_data_points = np.random.rand(100, 40).astype(np.float32)  # 100 new points
nmslib_ann.add_items(new_data_points)
nmslib_ann.build_index(np.vstack([data_points, new_data_points]))
logger.info("Index updated with new data points and rebuilt.")

# Benchmark the performance of querying
performance = nmslib_ann.benchmark_performance(query_points, k=5, rounds=3)
logger.info(f"Performance benchmark completed: {performance} seconds per query on average.")

# Save and load the index
nmslib_ann.save_index('nmslib_index.bin')
nmslib_ann.load_index('nmslib_index.bin')
logger.info("Index saved to 'nmslib_index.bin' and reloaded.")

# Print some complex query results
print("Sample Complex Query Results:", complex_results[:2])
print("Performance Benchmark:", performance)

# Finalize logging
logger.info("All operations completed successfully.")
