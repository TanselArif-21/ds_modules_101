import pkg_resources
import pandas as pd

titanic_f = pkg_resources.resource_filename(__name__, 'titanic/titanic.csv')
titanic_df = pd.read_csv(titanic_f)