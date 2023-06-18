import streamlit as st

st.title("Data")

st.markdown("""
Data are coming from a Kaggle hackathon, provided by the United States Geological Survey.

* Data set source: https://www.kaggle.com/competitions/model-the-impossible-predicting-ph-earthquakes

Data have been cleaned into homogeneous magnitude type and having more consistent values, so the data set has been reduced in dimensions (columns and rows) 
and time horizon.

Variables: from 9 to 5

Rows: from 10.188 to 8.999

Time horizion: from 1979-2020 to 1980-2009

### Data Description

#### Variables

**time**

Index of the time series. Time when the event occurred. 

**latitude**

Decimal degrees latitude.

**longitude**

Decimal degrees longitude.

**depth**

The depth where the earthquake begins to rupture.

**magType**

The method or algorithm used to calculate the preferred magnitude for the event.

**place**

Textual description of named geographic region near to the event. This may be a city name, or a Flinn-Engdahl Region name.

**type**

Type of seismic event.

**locationSource**

The network that originally authored the reported location of this event.

**magSource**

Network that originally authored the reported magnitude for this event.

**mag**

Response variable. The magnitude for the event.

#### Evaluation Metric

The evaluation metric used for this data set are MAE and RMSE scores

""")
