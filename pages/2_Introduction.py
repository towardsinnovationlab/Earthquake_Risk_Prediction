import streamlit as st

st.title("Introduction")

st.write("worldatlas")
st.image("./images/earthquake-cause.png", width=500)


st.markdown("""

An earthquake is a natural event that happen when the ground shakes and moves. It is caused by the movement of tectonic plates along faults 
or fractures in the Earth’s crust when they collide or move against each other.
An earthquake can release a large amount of energy in the form of seismic waves, which can travel through the Earth and cause damage to buildings, 
infrastructure, and human lives.
By understanding the causes and effects of earthquakes, insurers can better assess the risks associated with insuring properties located in areas 
prone to seismic activity. They can then use this knowledge to set appropriate premiums for their customers, ensuring that those who live in 
high-risk areas pay higher rates while those who live in low-risk areas pay lower rates.

Given an historical time series of Philippines earthquakes from 1979 to 2010, the purpose of this job is to predict the magnitude of 
the next earthquakes in the last year by the help of Supervised Learning models.

Magnitude variable is the outcome and it can be used as a feature in a pricing database to improve the quote of the risk premium.

When we use GLM in a pricing, we estimate the conditional mean of the response variable given the predictor variables. 
In estimating hazard events like earthquakes become relevant to have a measure of the prediction uncertainty, and the quantile regression solves 
this issue replacing a single value prediction by prediction intervals. 
Quantile regression allows for flexible modelling of the entire distribution of the response variable, it estimates the conditional quantiles 
of the response variable, given the explanatory variables.

For an overall overview of the analysis look the [repository](https://github.com/claudio1975/Earthquakes_Risk_Modelling)
""")
