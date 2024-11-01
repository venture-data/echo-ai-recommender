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
        train_t = sparse.csr_matrix(train_matrix.T)
        # print("Fitting the model...")
        self.model.fit(train_t)
        print("Model training complete.")

    def user_recommend(self, user_id, test_matrix, reverse_product_map, user_idx_dict, N=10):
        """
        Generate product recommendations for a specified user.

        Parameters:
        - user_id (int): The ID of the user for whom to generate recommendations.
        - test_matrix (csr_matrix): Sparse matrix for test data (item-user format).
        - reverse_product_map (dict): Dictionary mapping product indices to original product IDs.
        - orders_df (DataFrame): DataFrame containing user_id and corresponding user_idx mappings.
        - N (int): Number of recommendations to generate.

        Returns:
        - recommended_products_df (DataFrame): DataFrame with columns 'product_id' and 'score'.
        """
        # Step 1: Get the user index (user_idx) from user_id
        try:
            user_idx = user_idx_dict[user_id]
        except KeyError:
            print(f"User ID {user_id} not found in user_idx_dict.")
            return

        # Step 2: Retrieve the user's interaction row and convert to CSR format
        user_interactions = sparse.csr_matrix(test_matrix[user_idx])

        # Step 3: Check if the user has any interactions
        if user_interactions.nnz == 0:
            print(f"No interactions found for user {user_id}.")
            return

        # Step 4: Generate recommendations for the user using model.recommend
        recommended_products, scores = self.model.recommend(user_idx, user_interactions, N=N)

        # Step 5: Map recommended product indices to original product IDs and scores
        recommendations = [
            {'product_id': reverse_product_map.get(idx), 'score': score}
            for idx, score in zip(recommended_products, scores)
            if reverse_product_map.get(idx) is not None
        ]

        # Step 6: Convert the list of recommendations to a DataFrame
        # recommended_products_df = pd.DataFrame(recommendations)

        return recommendations
