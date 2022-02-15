# preload dataset automatically, if not already in place.
import os
from pathlib import Path
import numpy as np
import pandas as pd

# def download_file_from_gdrive(file_url, filename, out_path: Path, overwrite=False):
#     """
#     Downloads a file from GDrive given an URL
#     :param file_url: a string formated as https://drive.google.com/uc?id=<file_id>
#     :param: the desired file name
#     :param: the desired folder where the file will be downloaded to
#     :param overwrite: whether to overwrite the file if it already exists
#     """
#     file_exists = os.path.exists(f'{out_path}/{filename}')

#     if (file_exists and overwrite) or (not file_exists):
#         os.system(f'gdown {file_url} -O {out_path}/{filename}')

# FILE_URL = "https://drive.google.com/uc?id=1_lqydkMrmyNAgG4vU4wVmp6-j7tV0XI8"
# FILE_NAME = "renthop_train.json.gz"
# DATA_PATH = Path("../../_static/data/")

# download_file_from_gdrive(file_url=FILE_URL, filename= FILE_NAME, out_path=DATA_PATH)



df = pd.read_json(Path("../Data_OMLC/renthop_train.json.gz"), compression="gzip",
                  convert_dates=['created'])
print(df.info())


import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import beta, shapiro
from sklearn.preprocessing import StandardScaler


from sklearn.preprocessing import MinMaxScaler

# Let's draw plots!
import statsmodels.api as sm

# Let's take the price feature from Renthop dataset and filter by hands the most extreme values for clarity

price = df.price[(df.price <= 20000) & (df.price > 500)]
price_log = np.log(price)

# A lot of gestures so that sklearn didn't shower us with warnings
price_mm = (
    MinMaxScaler()
    .fit_transform(price.values.reshape(-1, 1).astype(np.float64))
    .flatten()
)
price_z = (
    StandardScaler()
    .fit_transform(price.values.reshape(-1, 1).astype(np.float64))
    .flatten()
)

# plt.subplots(sharey = True, figsize = (12, 10))
sm.qqplot(price, loc=price.mean(), scale=price.std())
sm.qqplot(price_z, loc=price_z.mean(), scale=price_z.std())


sm.qqplot(price_log, loc=price_log.mean(), scale=price_log.std())

plt.show()