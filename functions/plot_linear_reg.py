import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def plot_linear_reg(yfData, path_save):

  ## clean data
  yfData.reset_index(inplace=True)
  yfData.reset_index(inplace=True)

  ## linear regression
  x = yfData["index"]
  date = yfData["Date"]

  y = np.log10(yfData["Close"])

  slope, intercept = np.polyfit(x, y, 1)  # linear model adjustment

  y_model = np.polyval([slope, intercept], x)   # modeling...

  x_mean = np.mean(x)
  y_mean = np.mean(y)
  n = x.size                        # number of samples
  m = 2                             # number of parameters
  dof = n - m                       # degrees of freedom
  t = stats.t.ppf(0.975, dof)       # Students statistic of interval confidence

  residual = y - y_model

  std_error = (np.sum(residual**2) / dof)**.5   # Standard deviation of the error

  # calculating the r2
  # https://www.statisticshowto.com/probability-and-statistics/coefficient-of-determination-r-squared/
  # Pearson's correlation coefficient
  numerator = np.sum((x - x_mean)*(y - y_mean))
  denominator = ( np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2) )**.5
  correlation_coef = numerator / denominator
  r2 = correlation_coef**2

  # mean squared error
  MSE = 1/n * np.sum( (y - y_model)**2 )

  # to plot the adjusted model
  y_line = np.polyval([slope, intercept], x)

  # confidence interval
  # ci = t * std_error * (1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))**.5
  ci = std_error

  # predicting interval
  #pi = t * std_error * (1 + 1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))**.5

  ############### Ploting
  plt.rcParams.update({'font.size': 14})
  fig = plt.figure()
  ax = fig.add_axes([.1, .1, .8, .8])

  ax.plot(date, np.power(10, y), color = '#CC3F0C')
  ax.plot(date, np.power(10, y_line), color = '#ECA400')
  ax.fill_between(date, np.power(10, y_line + 2*ci), np.power(10,y_line - 2*ci), color = '#35E0E9', label = '2 sd')
  ax.fill_between(date, np.power(10, y_line + ci), np.power(10,y_line - ci), color = '#0A9DAE', label = '1 sd')

  ax.set_xlabel('Time')
  ax.set_ylabel('Close')

  # rounding and position must be changed for each case and preference
  a = str(np.round(intercept))
  b = str(np.round(slope,2))
  r2s = str(np.round(r2,2))
  MSEs = str(np.round(MSE))

  max_y = np.power(10, y.max())
  min_date = date.min()

  ax.text(min_date, max_y, 'log10(Close) = ' + a + ' + ' + b + ' Time')
  ax.text(min_date, max_y/10, '$r^2$ = ' + r2s + '     MSE = ' + MSEs)

  plt.legend(bbox_to_anchor=(1, .25), fontsize=12)
  plt.yscale("log")
  plt.savefig(path_save)
  plt.show()
