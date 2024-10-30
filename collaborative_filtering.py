import implicit
from scipy import sparse

class CollaborativeFiltering:
    def __init__(self, factors=2000, regularization=0.1, iterations=20):
        """
        Initialize the Collaborative Filtering model with ALS.

        Parameters:
        - factors (int): The number of latent factors to use.
        - regularization (float): Regularization parameter to avoid overfitting.
        - iterations (int): Number of ALS iterations to run.
        """
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations
        )

    def train(self, train_matrix):
        """
        Train the ALS model using the provided training matrix.

        Parameters:
        - train_matrix (csr_matrix): The user-item interaction matrix (in item-user format).
        """
        # Fit the model on the transpose of the training matrix
        self.model.fit(train_matrix.T)
        print("Model training complete.")

    def recommend(self, user_id, test_matrix, reverse_product_map, N=10):
        """
        Generate product recommendations for a specified user.

        Parameters:
        - user_id (int): The ID of the user for whom to generate recommendations.
        - test_matrix (csr_matrix): Sparse matrix for test data (item-user format).
        - reverse_product_map (dict): Dictionary mapping product indices to original product IDs.
        - N (int): Number of recommendations to generate.

        Returns:
        - recommended_product_ids (list): List of recommended product IDs.
        """
        # Transpose the test_matrix to get user-item format
        test_matrix_user_item = test_matrix.T

        # Get the user's interaction history
        user_interactions = test_matrix_user_item[user_id]
        
        # Check if the user has any interactions
        if user_interactions.nnz == 0:
            print(f"No interactions found for user {user_id}.")
            return []

        # Convert user interactions to csr format (for ALS model)
        user_interactions_csr = sparse.csr_matrix(user_interactions)

        # Generate recommendations for the user
        recommended_products, _ = self.model.recommend(user_id, user_interactions_csr, N=N)
        
        # Map recommended product indices to original product IDs
        recommended_product_ids = [reverse_product_map.get(idx, None) for idx in recommended_products]
        
        # Filter out None values (in case some indices were missing in the map)
        recommended_product_ids = [pid for pid in recommended_product_ids if pid is not None]
        
        return recommended_product_ids
