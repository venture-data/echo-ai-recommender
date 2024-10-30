# src/frequently_bought_together.py
import os
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
import polars as pl
from scipy.sparse import csr_matrix
import pickle

class FrequentlyBoughtTogether:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.models_dir = 'models'
        os.makedirs(self.models_dir, exist_ok=True)

    def _collect_unique_products(self):
        return (
            pl.scan_csv(self.csv_path)
            .select(pl.col("product_name").unique())
            .collect()
            .to_series()
            .to_list()
        )

    def _create_product_index_map(self, unique_products):
        return {product: index for index, product in enumerate(unique_products)}

    def _load_and_group_dataset(self):
        df = pl.scan_csv(self.csv_path).collect()
        return df.group_by("order_id").agg(pl.col("product_name").alias("products"))

    def _process_transactions(self, df_grouped, product_index_map):
        row_indices, col_indices, data = [], [], []
        
        for transaction_idx, transaction in enumerate(df_grouped["products"]):
            for product in transaction:
                if product in product_index_map:
                    row_indices.append(transaction_idx)
                    col_indices.append(product_index_map[product])
                    data.append(1)

        return csr_matrix((data, (row_indices, col_indices)), 
                         shape=(len(df_grouped), len(product_index_map)))

    def train(self, min_support=0.0001, min_confidence=0.1):
        print("Processing dataset...")
        unique_products = self._collect_unique_products()
        product_index_map = self._create_product_index_map(unique_products)
        df_grouped = self._load_and_group_dataset()
        sparse_matrix = self._process_transactions(df_grouped, product_index_map)
        
        print("Training model...")
        df_trans = pd.DataFrame.sparse.from_spmatrix(sparse_matrix, columns=unique_products)
        frequent_itemsets = fpgrowth(df_trans, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        
        # Save models
        with open(os.path.join(self.models_dir, 'frequent_itemsets.pkl'), 'wb') as f:
            pickle.dump(frequent_itemsets, f)
        with open(os.path.join(self.models_dir, 'association_rules.pkl'), 'wb') as f:
            pickle.dump(rules, f)
            
        print("Model training completed!")
        return frequent_itemsets, rules
    
    def load_model(self, frequent_itemsets_path, association_rules_path):
        """
        Load the saved frequent itemsets and association rules from pickle files.

        Args:
        frequent_itemsets_path (str): Path to the saved frequent itemsets pickle file.
        association_rules_path (str): Path to the saved association rules pickle file.

        Returns:
        tuple: (frequent_itemsets, rules)
        """
        with open(os.path.join(self.models_dir, frequent_itemsets_path), 'rb') as f:
            frequent_itemsets = pickle.load(f)

        with open(os.path.join(self.models_dir, association_rules_path), 'rb') as f:
            rules = pickle.load(f)

        return frequent_itemsets, rules

    def get_recommendations(self, cart_items, rules, top_n=3):
        
        cart_items_set = frozenset(cart_items)
        recommendations = rules[rules['antecedents'].apply(lambda x: cart_items_set.issubset(x))]
        return recommendations[['consequents', 'confidence', 'lift']].sort_values(
            by='confidence', ascending=False).head(top_n)

# main.py
def main():
    csv_path = "Merged_Dataset.csv"
    frequent_itemsets_path = 'frequent_itemsets.pkl'
    association_rules_path = 'association_rules.pkl'
    fbt = FrequentlyBoughtTogether(csv_path)
    # _, rules = fbt.train()
    _, rules = fbt.load_model(frequent_itemsets_path, association_rules_path)
    
    cart_items = ['Soda']
    recommendations = fbt.get_recommendations(cart_items, rules)
    print(f"\nRecommendations for {cart_items}:")
    print(recommendations)

if __name__ == "__main__":
    main()