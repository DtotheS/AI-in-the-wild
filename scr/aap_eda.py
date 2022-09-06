import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime as dt
import csv
import seaborn as sns

df = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/aap_090122.csv")
df.columns