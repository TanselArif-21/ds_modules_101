import pkg_resources
import pandas as pd

titanic_f = pkg_resources.resource_filename(__name__, 'titanic/titanic.csv')
titanic_df = pd.read_csv(titanic_f)

ibd_f = pkg_resources.resource_filename(__name__, 'ibd/IBD.csv')
ibd_df = pd.read_csv(ibd_f)

hr_f = pkg_resources.resource_filename(__name__, 'HR/HR.csv')
hr_df = pd.read_csv(hr_f)