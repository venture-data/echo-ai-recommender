import pandas as pd

class Top20Prod:
    def __init__(self, dataframe):
        """
        Initialize the Top20Prod with a DataFrame.

        Parameters:
        dataframe (DataFrame): The dataset containing product and order information.
        """
        self.dataframe = dataframe

    def get_top_20_products(self):
        """
        Return the top 20 products across all aisles.

        Returns:
        DataFrame: Top 20 products from the entire dataset.
        """
        # Grouping by 'product_name' to calculate the number of orders for each product
        top_items = self.dataframe.groupby('product_name').size().reset_index(name='count')

        # Finding the top 20 products for the entire dataset
        top_20_products = top_items.nlargest(20, 'count').reset_index(drop=True)

        return top_20_products

    @classmethod
    def load_from_pickle(cls, filename):
        """
        Load the DataFrame from a pickle file.

        Parameters:
        filename (str): The name of the pickle file to load.

        Returns:
        Top20Prod: An instance of ProductAnalyzer with the loaded DataFrame.
        """
        dataframe = pd.read_pickle(filename)
        return cls(dataframe)

