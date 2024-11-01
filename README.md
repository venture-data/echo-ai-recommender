# echo-ai-recommender

## CollaborativeFiltering Class

This class, found in `collaborative_filtering.py`, implements collaborative filtering using Alternating Least Squares (ALS) with `implicit`. It provides methods for splitting interaction data, training the model, and generating recommendations.

#### Class and Method Descriptions

#### `__init__(self, factors=2000, regularization=0.1, iterations=20)`
Initializes the ALS model with configurable parameters:
- **factors**: Number of latent factors for the model (default: 2000).
- **regularization**: Regularization parameter to prevent overfitting (default: 0.1).
- **iterations**: Number of ALS iterations for model training (default: 20).

#### `_validate_train_size(train_size: float) -> None`
Static method that checks if `train_size` is a float between 0 and 1. Used to validate input for train-test split ratio.

#### `_get_stratified_tr_mask(user_idx, product_idx, train_size, random_state)`
Creates a stratified mask for train-test splitting, ensuring each user and item appears in the training set:
- **user_idx**: Array of user indices.
- **product_idx**: Array of product indices.
- **train_size**: Proportion of data for training.
- **random_state**: Seed for reproducibility.

#### `_make_sparse_tr_te(user_idx, product_idx, interaction, train_mask)`
Generates sparse matrices for training and testing:
- **user_idx, product_idx**: Arrays of indices.
- **interaction**: Array of interaction values.
- **train_mask**: Boolean mask for train-test split.

Returns:
- **train_interaction_matrix**: Sparse matrix for training.
- **test_interaction_matrix**: Sparse matrix for testing.

#### `train_test_split(self, interaction_df: pl.DataFrame, train_size=0.75, random_state=None) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]`
Splits a `polars.DataFrame` of interactions into training and test sets:
- **interaction_df**: DataFrame with columns `user_idx`, `product_idx`, `interaction`.
- **train_size**: Ratio for train-test split (default: 0.75).
- **random_state**: Seed for reproducibility.

Returns:
- **train_matrix** and **test_matrix** as sparse matrices.

#### `train(self, interaction_input: Union[sparse.csr_matrix, pl.DataFrame], is_csr=True, train_size=0.75, random_state=None) -> Union[None, Tuple[sparse.csr_matrix, sparse.csr_matrix]]`
Trains the ALS model on either a CSR matrix or a `polars.DataFrame` of interactions.
- **interaction_input**: Interaction data (CSR matrix or DataFrame).
- **is_csr**: Flag for direct CSR matrix training (default: True).
- **train_size**: Ratio for train-test split if using a DataFrame.
- **random_state**: Seed for reproducibility.

Returns:
- **train_matrix** and **test_matrix** if `interaction_input` is a DataFrame; otherwise, `None`.

#### `user_recommend(self, user_id, test_matrix, reverse_product_map, user_idx_dict, N=10)`
Generates product recommendations for a given user:
- **user_id**: Target user’s ID.
- **test_matrix**: Sparse matrix in item-user format.
- **reverse_product_map**: Dictionary mapping product indices to product IDs.
- **user_idx_dict**: Dictionary mapping user IDs to indices.
- **N**: Number of recommendations to return (default: 10).

Returns:
- A list of top `N` recommended products, each with `product_id` and `score`.

### Example Usage

```python
import polars as pl
from collaborative_filtering import CollaborativeFiltering

# Load interaction data into a polars DataFrame
interaction_df = pl.read_csv('/path/to/interactions.csv')

# Initialize the CollaborativeFiltering model
cf = CollaborativeFiltering(factors=100, regularization=0.1, iterations=15)

# Train the model using interaction_df as input
train_matrix, test_matrix = cf.train(interaction_input=interaction_df, is_csr=False, train_size=0.8, random_state=42)

# Create mappings for user indices and product indices
user_idx_dict = {row['user_id']: row['user_idx'] for row in interaction_df.select(['user_id', 'user_idx']).to_dicts()}
reverse_product_map = {row['product_idx']: row['product_id'] for row in interaction_df.select(['product_id', 'product_idx']).to_dicts()}

# Generate recommendations for a specific user
user_id = 1  # Example user ID
recommendations = cf.user_recommend(user_id, test_matrix, reverse_product_map, user_idx_dict, N=5)

print("Recommendations for user", user_id, ":", recommendations)
```

