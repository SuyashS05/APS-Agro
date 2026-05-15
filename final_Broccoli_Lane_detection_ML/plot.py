from matplotlib import pyplot as plt
import pandas as pd

df=pd.read_excel("Gps module/book3.xlsx")

plt.xlabel("Latitude")
plt.ylabel("Longitude")

plt.plot(df["Latitude"],df["Longitude"])
plt.title("GPS Path")
plt.show()