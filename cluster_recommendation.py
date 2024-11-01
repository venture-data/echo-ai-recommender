import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from scipy.spatial.distance import cdist

class ClusterRecommendation:
    def __init__(self, csv_file, pickle_file):
        """
        Initialize the ClusterRecommendation class.

        Parameters:
        csv_file (str): The path to the CSV file containing cluster recommendations.
        pickle_file (str): The path to the pickle file containing user_id and cluster information.
        """ 
        self.recommendations_df = pd.read_csv(csv_file)
        self.user_cluster_df = pd.read_pickle(pickle_file)

    def train_model(self, merged_orders):
        """
        Train the model to generate the cluster recommendation files.

        Parameters:
        merged_orders (DataFrame): The DataFrame containing user_id and aisle_id information.
        """
        # Create the user-aisle crosstab
        data_user_aisle = pd.crosstab(merged_orders['user_id'], merged_orders['aisle_id'])
        
        # PCA Transformation
        pca = PCA(n_components=44)
        ps = pca.fit_transform(data_user_aisle)

        # Shuffle and randomly sample 10,000 rows
        X_random_sample = shuffle(data_user_aisle, random_state=42).iloc[:10000, :]
        
        # KMeans Clustering
        clusterer = KMeans(n_clusters=3, random_state=42)
        c_preds = clusterer.fit_predict(ps)

        # Compute distances between items and cluster centers
        centers = clusterer.cluster_centers_
        distances = cdist(ps, centers, 'euclidean')

        # Create a mapping of product IDs to product names from the merged_orders DataFrame
        product_id_to_name = dict(zip(merged_orders['product_id'], merged_orders['product_name']))

        # For each cluster, find the top 10 closest items to the center
        closest_items = {}
        
        for cluster_id in range(centers.shape[0]):
            cluster_distances = distances[c_preds == cluster_id, cluster_id]
            closest_items_indices = np.argsort(cluster_distances)[:10]
            closest_items[f'Cluster_{cluster_id}'] = [product_id_to_name[merged_orders['product_id'].iloc[item]] for item in closest_items_indices]

        # Create a DataFrame to hold the closest items with product names
        closest_items_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in closest_items.items()]))
        
        # Save the DataFrame to a CSV file
        closest_items_df.to_csv("cluster_products2.csv", index=False)

        # Create user-cluster mapping
        merged_orders['cluster'] = [c_preds[user_id - 1] for user_id in merged_orders['user_id']]
        distinct_clusters = merged_orders[['user_id', 'cluster']].drop_duplicates()
        
        # Saving the result in a pickle file
        distinct_clusters.to_pickle("user_clusters2.pkl")

        
    def get_user_cluster(self, user_id):
        """
        Get the cluster for a given user ID.

        Parameters:
        user_id (int): The user ID to look up.

        Returns:
        int: The cluster ID the user belongs to, or None if not found.
        """
        user_cluster = self.user_cluster_df[self.user_cluster_df['user_id'] == user_id]
        if not user_cluster.empty:
            return user_cluster['cluster'].values[0]
        return None

    def get_recommendations(self, user_id):
        """
        Get product recommendations for a given user ID.

        Parameters:
        user_id (int): The user ID for which to fetch recommendations.

        Returns:
        tuple: A tuple containing the cluster ID and a list of recommended products, 
               or a message if the user is not found.
        """
        cluster_id = self.get_user_cluster(user_id)
        
        if cluster_id is not None:
            # Fetch the top 10 recommendations for the user's cluster
            recommended_products = self.recommendations_df[f'Cluster_{cluster_id}'].head(10).tolist()
            return cluster_id, recommended_products
        else:
            return None, f"No recommendations found for user ID {user_id}."
