import implicit
from scipy import sparse
import numpy as np
from typing import Optional, Tuple, Union, Dict, List
from sklearn.utils import check_random_state
import polars as pl

class CollaborativeFiltering:
    def __init__(self, factors: int = 2000, regularization: float = 0.1, iterations: int = 20):
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

    @staticmethod
    def _validate_train_size(train_size: float) -> None:
        assert isinstance(train_size, float) and 0. < train_size < 1., \
            "train_size should be a float between 0 and 1"

    @staticmethod
    def _get_stratified_tr_mask(user_idx, product_idx, train_size, random_state):
        CollaborativeFiltering._validate_train_size(train_size)
        random_state = check_random_state(random_state)
        n_interactions = user_idx.shape[0]

        # Create a random mask for training data
        train_mask = random_state.rand(n_interactions) <= train_size

        # Ensure at least one of each user and product is in the training set
        for array in (user_idx, product_idx):
            present = array[train_mask]
            missing = np.unique(array[~train_mask][~np.isin(array[~train_mask], present)])

            if missing.shape[0] == 0:
                continue

            array_mask_missing = np.isin(array, missing)
            where_missing = np.where(array_mask_missing)[0]

            added = set()
            for idx, val in zip(where_missing, array[where_missing]):
                if val in added:
                    continue
                train_mask[idx] = True
                added.add(val)

        return train_mask

    @staticmethod
    def _make_sparse_tr_te(user_idx, product_idx, interaction, train_mask):
        train_interaction_matrix = sparse.csr_matrix((interaction[train_mask], (product_idx[train_mask], user_idx[train_mask])))
        test_interaction_matrix = sparse.csr_matrix((interaction, (product_idx, user_idx)))
        return train_interaction_matrix, test_interaction_matrix

    def train_test_split(self, interaction_df: pl.DataFrame, train_size: float = 0.75, random_state: Optional[int] = None) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """
        Split the interaction DataFrame into train and test matrices.
        """
        user_idx = interaction_df['user_idx'].to_numpy()
        product_idx = interaction_df['product_idx'].to_numpy()
        interaction = interaction_df['interaction'].cast(float).to_numpy()

        train_mask = self._get_stratified_tr_mask(user_idx, product_idx, train_size=train_size, random_state=random_state)

        # Create sparse matrices for the training and test sets
        train_matrix, test_matrix = self._make_sparse_tr_te(user_idx, product_idx, interaction, train_mask)
        return train_matrix, test_matrix

    def train(
        self, 
        interaction_input: Union[sparse.csr_matrix, pl.DataFrame], 
        is_csr: bool = True, 
        train_size: float = 0.75, 
        random_state: Optional[int] = None
    ) -> Union[None, Tuple[sparse.csr_matrix, sparse.csr_matrix]]:
        """
        Train the ALS model using either a precomputed CSR matrix or an interaction DataFrame.

        Parameters:
        - interaction_input: Either a CSR matrix or a DataFrame with user and product interactions.
        - is_csr (bool): Flag indicating if interaction_input is already a CSR matrix.
        - train_size (float): Train-test split ratio if using a DataFrame as input.
        - random_state (Optional[int]): Seed for reproducibility.

        Returns:
        - Tuple of train_matrix and test_matrix if interaction_input is a DataFrame, None otherwise.
        """
        if is_csr:
            if not sparse.isspmatrix_csr(interaction_input):
                raise ValueError("If is_csr is True, interaction_input must be a CSR sparse matrix.")
            
            # Fit the model directly on the provided CSR matrix
            train_t = sparse.csr_matrix(interaction_input.T)
            self.model.fit(train_t)
            print("Model training complete with direct CSR matrix.")
            return None
        else:
            if not isinstance(interaction_input, pl.DataFrame):
                raise ValueError("If is_csr is False, interaction_input must be a polars DataFrame.")

            # Split the DataFrame into train and test CSR matrices
            train_matrix, test_matrix = self.train_test_split(interaction_input, train_size, random_state)
            
            # Train on the transposed train matrix
            train_t = sparse.csr_matrix(train_matrix.T)
            self.model.fit(train_t)
            print("Model training complete with generated train matrix from DataFrame.")
            return train_matrix, test_matrix

    def user_recommend(self, user_id, test_matrix, reverse_product_map, user_idx_dict, N=10):
        """
        Generate product recommendations for a specified user.

        Parameters:
        - user_id (int): The ID of the user for whom to generate recommendations.
        - test_matrix (csr_matrix): Sparse matrix for test data (item-user format).
        - reverse_product_map (dict): Dictionary mapping product indices to original product IDs.
        - user_idx_dict (dict): Dictionary mapping user IDs to user indices.
        - N (int): Number of recommendations to generate.

        Returns:
        - List of recommended products with 'product_id' and 'score'.
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

        return recommendations