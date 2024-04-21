import nmslib
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("NmslibDirectExample")

# Initialize and build an NMSLIB index
def create_and_build_nmslib_index(space='cosinesimil', method='hnsw', dtype=nmslib.DistType.FLOAT, data_points=None):
    index = nmslib.init(method=method, space=space, dtype=dtype)
    if data_points is not None:
        for i, vector in enumerate(data_points):
            index.addDataPoint(i, vector)
        index.createIndex({'M': 30, 'post': 0, 'efConstruction': 100}, print_progress=False)
    logger.info(f"Initialized and built NMSLIB index with space {space}, method {method}, and data type {dtype}.")
    return index

# Generate random high-dimensional data points
np.random.seed(42)
data_points = np.random.rand(1000, 40).astype(np.float32)  # 1000 points in 40 dimensions
query_points = np.random.rand(10, 40).astype(np.float32)  # 10 query points

# Initialize and build the NMSLIB index
index = create_and_build_nmslib_index(data_points=data_points)

# Perform a batch of complex queries
complex_results = []
for point in query_points:
    ids, distances = index.knnQuery(point, k=5)
    complex_results.append(ids)
logger.info("Performed batch querying.")

# Simulate an update by creating a new index with additional data points
new_data_points = np.random.rand(100, 40).astype(np.float32)  # 100 new points
all_data_points = np.vstack([data_points, new_data_points])
index = create_and_build_nmslib_index(data_points=all_data_points)
logger.info("Index rebuilt with additional data points.")

# Benchmark the performance of querying
import time
start_time = time.time()
for _ in range(3):  # Rounds
    for query in query_points:
        index.knnQuery(query, k=5)
end_time = time.time()
performance = (end_time - start_time) / (len(query_points) * 3)
logger.info(f"Performance benchmark completed: {performance} seconds per query on average.")

# Save and load the index
index.saveIndex('nmslib_index.bin', save_data=True)
index.loadIndex('nmslib_index.bin', load_data=True)
logger.info("Index saved to 'nmslib_index.bin' and reloaded.")

# Print some of the complex query results
print("Sample Complex Query Results:", complex_results[:2])
print("Performance Benchmark:", performance)

# Finalize logging
logger.info("All operations completed successfully.")
