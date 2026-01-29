import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Call Center Forecasting", layout="centered")

st.title("📞 Call Center Workload & Sentiment Forecasting")

st.write("""
This system predicts future call volume and customer sentiment trends 
to help call centers plan staffing and reduce losses.
""")

# Dummy historical data
days = np.arange(1, 31)
call_volume = np.random.randint(200, 500, size=30)
sentiment_score = np.random.uniform(-1, 1, size=30)

data = pd.DataFrame({
    "Day": days,
    "Call Volume": call_volume,
    "Sentiment Score": sentiment_score
})

st.subheader("📊 Historical Call Center Data")
st.dataframe(data)

# Forecast call volume using linear regression
X = data[["Day"]]
y = data["Call Volume"]
model = LinearRegression()
model.fit(X, y)

future_days = np.arange(31, 38).reshape(-1, 1)
predicted_calls = model.predict(future_days)

st.subheader("📈 Future Call Volume Prediction (Next 7 Days)")
fig, ax = plt.subplots()
ax.plot(data["Day"], data["Call Volume"], label="Historical Calls")
ax.plot(future_days, predicted_calls, '--', label="Predicted Calls")
ax.set_xlabel("Day")
ax.set_ylabel("Number of Calls")
ax.legend()
st.pyplot(fig)

# Sentiment trend
st.subheader("😊 Customer Sentiment Trend")
fig2, ax2 = plt.subplots()
ax2.plot(data["Day"], data["Sentiment Score"])
ax2.set_xlabel("Day")
ax2.set_ylabel("Sentiment Score (-1 to +1)")
st.pyplot(fig2)

st.success("✅ Forecast helps optimize staffing and reduce customer dissatisfaction.")