import implicit
from scipy import sparse
import numpy as np
from typing import Optional, Tuple, Union
from sklearn.utils import check_random_state
import polars as pl

class ContentbBasedFiltering:
    def __init__(self, factors: int = 2000, regularization: float = 0.1, iterations: int = 20):
        """
        Initialize the Content Based Filtering model with ALS.
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
        ContentbBasedFiltering._validate_train_size(train_size)
        random_state = check_random_state(random_state)
        n_interactions = user_idx.shape[0]

        train_mask = random_state.rand(n_interactions) <= train_size

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
        """
        if is_csr:
            if not sparse.isspmatrix_csr(interaction_input):
                raise ValueError("If is_csr is True, interaction_input must be a CSR sparse matrix.")
            
            train_t = sparse.csr_matrix(interaction_input.T)
            self.model.fit(train_t)
            print("Model training complete with direct CSR matrix.")
            return None
        else:
            if not isinstance(interaction_input, pl.DataFrame):
                raise ValueError("If is_csr is False, interaction_input must be a polars DataFrame.")

            train_matrix, test_matrix = self.train_test_split(interaction_input, train_size, random_state)
            train_t = sparse.csr_matrix(train_matrix.T)
            self.model.fit(train_t)
            print("Model training complete with generated train matrix from DataFrame.")
            return train_matrix, test_matrix

    def product_recommend(self, product_id, reverse_product_map, product_id_to_idx, n_similar=10):
        """
        Find products similar to the given product ID.
        
        Parameters:
        - product_id: The ID of the product to find similar products for.
        - reverse_product_map: Dictionary mapping product indices to product IDs.
        - product_id_to_idx: Dictionary mapping product IDs to product indices.
        - n_similar: Number of similar products to retrieve (default: 10).

        Returns:
        - A list of dictionaries with 'product_id' and 'similarity_score' for similar products.
        """
        # Step 1: Get the product index using product_id_to_idx
        product_idx = product_id_to_idx.get(product_id)
        if product_idx is None:
            print(f"Product ID {product_id} not found in product_id_to_idx.")
            return []

        # Step 2: Use the ALS model to find similar items
        similar_products = self.model.similar_items(product_idx, n_similar)
        # display(similar_products)
        product_indices, scores = similar_products

        # Step 4: Create a list to collect similar products
        similar_products_list = []

        # Step 5: Iterate through the product indices and scores
        for idx, score in zip(product_indices, scores):
            original_product_id = reverse_product_map.get(idx)

            # Check if the product index exists in the reverse map
            if original_product_id is not None:
                similar_products_list.append({
                    'product_id': original_product_id,
                    'similarity_score': score
                })
            else:
                print(f"Product index {idx} not found in reverse_product_map.")

        return similar_products_list