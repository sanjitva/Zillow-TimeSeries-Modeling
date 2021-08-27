"""
This module contains functions employed during the EDA process and for repeated processes during model testing. It is
used in the main Jupyter Notebook with the import statement 'import tools.helpers as th'.

CONTENTS:

Imports
I. Diagnostic Functions
II. Transformation Functions
"""

import numpy as np
import pandas as pd



def melt_data(df):
        
    melted = pd.melt(df, id_vars=['RegionName', 'RegionID', 'SizeRank', 'City', 'State', 'Metro', 'CountyName'], var_name='time')
    melted['time'] = pd.to_datetime(melted['time'], infer_datetime_format=True)
    melted = melted.dropna(subset=['value'])
    return melted.groupby('time').aggregate({'value':'mean'})
