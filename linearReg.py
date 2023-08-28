import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

if __name__ == "__main__":
  data = pd.read_csv('Salary_Data.csv')
  X = data[['YearsExperience']]
  y = data['Salary']
  X = sm.add_constant(X)
  model = sm.OLS(y, X).fit()
  result = model.summary()
  print(result)
