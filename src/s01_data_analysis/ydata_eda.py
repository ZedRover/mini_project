import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport

df = pd.read_csv("data/data.csv", index_col=0)


profile = ProfileReport(df, title="Profiling Report", minimal=True)
profile.to_file("artifacts/eda_report.html")
