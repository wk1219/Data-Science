import matplotlib.pyplot as plt
import pandas as pd

covid_data = 'DataSet/Corona/covid_19_data.csv'
covid_csv = pd.read_csv(covid_data)

column_list = []

date = covid_csv['ObservationDate'].values.tolist()
confirmed = covid_csv['Confirmed'].values.tolist()

print("Row counts : %d" % len(covid_csv))

plt.bar(date, confirmed, color='red')
plt.ylim(0, 150000)

plt.xlabel("Date")
plt.ylabel("Confirmed")
plt.title("Global Corona Data")

plt.xticks(date[: :1000], confirmed[: :1000], rotation=45)

plt.show()