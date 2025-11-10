import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport

df = pd.read_csv("../data/data.csv", index_col=0)


profile = ProfileReport(df, title="Profiling Report")
profile.to_file("eda_report.html")
