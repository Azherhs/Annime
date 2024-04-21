from src.annoy_int import AnnoyANN
import numpy as np
import logging


# Function to create and build a new Annoy index
def create_and_build_annoy_index(data_points_ind, dim=100, metric='angular', num_trees=20):
    annoy_ann = AnnoyANN(dim=dim, metric=metric, num_trees=num_trees)
    annoy_ann.add_items(data_points_ind)
    annoy_ann.build_index(data_points_ind, num_trees=num_trees)
    return annoy_ann


# Initialize Annoy interface with angular metric and setup logging
np.random.seed(42)
data_points = np.random.rand(2000, 100)  # 2000 points in 100 dimensions
annoy_ann = create_and_build_annoy_index(data_points)

# Perform a batch of complex queries
query_points = np.random.rand(50, 100)  # 50 new random points
constraints = lambda x: np.linalg.norm(x) > 0.5  # Constraint: norm should be greater than 0.5
batch_results_with_constraints = [
    annoy_ann.query_with_constraints(point, constraints, k=10) for point in query_points
]
print("Batch Query Results with Constraints:", batch_results_with_constraints)

# Simulate item updating by recreating the index
new_vector = np.random.rand(100)
data_points[10] = new_vector  # Update the data point in the array
annoy_ann = create_and_build_annoy_index(data_points)  # Create and build a new index
updated_results = annoy_ann.query(new_vector, k=10)
print("Results after simulated update:", updated_results)

# Benchmark the performance of querying
benchmark_results = annoy_ann.benchmark_performance(query_points, k=10, rounds=5)
print("Performance Benchmark:", benchmark_results)

# Enable detailed logging and perform some operations
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
annoy_ann.enable_logging('DEBUG')
annoy_ann.optimize_index()  # Log an operation

# Save and load the index for demonstration of persistence
annoy_ann.save_index('final_annoy_index.ann')
annoy_ann.load_index('final_annoy_index.ann')

logger.info("Finished processing using Annoy interface")
