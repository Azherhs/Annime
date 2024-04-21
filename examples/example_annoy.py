from annoy import AnnoyIndex
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def create_and_build_annoy_index(data_points, dim=100, metric='angular', num_trees=20):
    index = AnnoyIndex(dim, metric)
    for i, vector in enumerate(data_points):
        index.add_item(i, vector)
    index.build(num_trees)
    return index

# Initialize data
np.random.seed(42)
data_points = np.random.rand(2000, 100)  # 2000 points in 100 dimensions

# Create and build Annoy index
annoy_index = create_and_build_annoy_index(data_points)

# Perform a batch of complex queries with constraints
query_points = np.random.rand(50, 100)  # 50 new random points
constraints = lambda x: np.linalg.norm(x) > 0.5  # Constraint: norm should be greater than 0.5
batch_results_with_constraints = []
for point in query_points:
    nns = annoy_index.get_nns_by_vector(point, 10, include_distances=True)
    constrained_results = [idx for idx in nns[0] if constraints(data_points[idx])]
    batch_results_with_constraints.append(constrained_results)
print("Batch Query Results with Constraints:", batch_results_with_constraints)

# Simulate item updating by recreating the index
new_vector = np.random.rand(100)
data_points[10] = new_vector  # Update the data point in the array
annoy_index = create_and_build_annoy_index(data_points)  # Create and build a new index
updated_results = annoy_index.get_nns_by_vector(new_vector, 10, include_distances=True)
print("Results after simulated update:", updated_results[0])

# Benchmark the performance of querying
import time
start_time = time.time()
for _ in range(5):  # Rounds
    for query in query_points:
        annoy_index.get_nns_by_vector(query, 10)
end_time = time.time()
print("Performance Benchmark:", (end_time - start_time) / (len(query_points) * 5), "seconds per query")

# Save and load the index for demonstration of persistence
annoy_index.save('final_annoy_index.ann')
annoy_index.load('final_annoy_index.ann')

logger.info("All operations completed successfully.")
