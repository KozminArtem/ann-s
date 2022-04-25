import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import matplotlib.pyplot as plt
#%config InlineBackend.figure_format = 'retina'


def download_file_from_gdrive(file_url, filename, out_path: Path, overwrite=False):
    """
    Downloads a file from GDrive given an URL
    :param file_url: a string formated as https://drive.google.com/uc?id=<file_id>
    :param: the desired file name
    :param: the desired folder where the file will be downloaded to
    :param overwrite: whether to overwrite the file if it already exists
    """
    file_exists = (out_path / filename).exists()

    if (file_exists and overwrite) or (not file_exists):
        os.system(f'gdown {file_url} -O {out_path}/{filename}')


FILE_URL = "https://drive.google.com/uc?id=1G3YjM6mR32iPnQ6O3f6rE9BVbhiTiLyU"
FILE_NAME = "medium_posts.csv"
DATA_PATH = Path("../Data_OMLC")

download_file_from_gdrive(file_url=FILE_URL, filename= FILE_NAME, out_path=DATA_PATH)

df = pd.read_csv(DATA_PATH / FILE_NAME,  sep="\t")


df = df[["published", "url"]].dropna().drop_duplicates()


df["published"] = pd.to_datetime(df["published"])


df.sort_values(by=["published"]).head(n=3)	

df = df[(df["published"] > "2012-08-15") & (df["published"] < "2017-06-26")].sort_values(by=["published"])

print(df.head(n=3))
print(df.tail(n=3))

aggr_df = df.groupby("published")[["url"]].count()
aggr_df.columns = ["posts"]
print(aggr_df.head(n=3))

daily_df = aggr_df.resample("D").apply(sum)
print(daily_df.head(n=3))


from plotly import graph_objs as go
from plotly.offline import init_notebook_mode, iplot, plot
from IPython.display import display, IFrame

# Initialize plotly
init_notebook_mode(connected=True)

def plotly_df(df, title="", width=800, height=500):
    """Visualize all the dataframe columns as line plots."""
    common_kw = dict(x=df.index, mode="lines")
    data = [go.Scatter(y=df[c], name=c, **common_kw) for c in df.columns]
    layout = dict(title=title)
    fig = dict(data=data, layout=layout)

    # in a Jupyter Notebook, the following should work
    #iplot(fig, show_link=False)

    # in a Jupyter Book, we save a plot offline and then render it with IFrame
    plot_path = f"../../_static/plotly_htmls/{title}.html".replace(" ", "_")
    plot(fig, filename=plot_path, show_link=False, auto_open=False);
    display(IFrame(plot_path, width=width, height=height))


plotly_df(daily_df, title="Posts on Medium (daily)")

plt.show()
