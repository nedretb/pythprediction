import pandas as pd
import numpy as np

from scipy import stats as sps
from scipy.interpolate import interp1d

from IPython.display import clear_output
from sqlalchemy import create_engine
import pymysql

import mysql.connector
from datetime import datetime, timedelta

sqlEngine = create_engine('mysql+pymysql://root:@127.0.0.1/test', pool_recycle=3600)

dbConnection = sqlEngine.connect()

states = pd.read_sql('select * from total_cases', index_col=['canton_id', 'date'], parse_dates=['date'], con=dbConnection).sort_index().squeeze()
dbConnection.close()
state_name = 0

def prepare_cases(cases, cutoff=25):
  new_cases = cases.diff()

  smoothed = new_cases.rolling(7,
                               win_type='gaussian',
                               min_periods=1,
                               center=True).mean(std=2).round()

  idx_start = np.searchsorted(smoothed, cutoff)

  smoothed = smoothed.iloc[idx_start:]
  original = new_cases.loc[smoothed.index]

  return original, smoothed


cases = states.xs(state_name).rename(f"{state_name} cases")



original, smoothed = prepare_cases(cases)



GAMMA = 4.8
R_T_MAX = 12
r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)


def highest_density_interval(pmf, p=.9, debug=False):
    # If we pass a DataFrame, just call this recursively on the columns
    if (isinstance(pmf, pd.DataFrame)):
        return pd.DataFrame([highest_density_interval(pmf[col], p=p) for col in pmf],
                            index=pmf.columns)

    cumsum = np.cumsum(pmf.values)

    # N x N matrix of total probability mass for each low, high
    total_p = cumsum - cumsum[:, None]

    # Return all indices with total_p > p
    lows, highs = (total_p > p).nonzero()

    # Find the smallest range (highest density)
    best = (highs - lows).argmin()

    low = pmf.index[lows[best]]
    high = pmf.index[highs[best]]

    return pd.Series([low, high],
                     index=[f'Low_{p * 100:.0f}',
                            f'High_{p * 100:.0f}'])


def get_posteriors(sr, sigma=2.3):
  # (1) Calculate Lambda
  lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))

  # (2) Calculate each day's likelihood
  likelihoods = pd.DataFrame(
    data=sps.poisson.pmf(sr[1:].values, lam),
    index=r_t_range,
    columns=sr.index[1:])

  # (3) Create the Gaussian Matrix
  process_matrix = sps.norm(loc=r_t_range,
                            scale=sigma
                            ).pdf(r_t_range[:, None])

  # (3a) Normalize all rows to sum to 1
  process_matrix /= process_matrix.sum(axis=0)

  # (4) Calculate the initial prior
  # prior0 = sps.gamma(a=4).pdf(r_t_range)
  prior0 = np.ones_like(r_t_range) / len(r_t_range)
  prior0 /= prior0.sum()

  # Create a DataFrame that will hold our posteriors for each day
  # Insert our prior as the first posterior.
  posteriors = pd.DataFrame(
    index=r_t_range,
    columns=sr.index,
    data={sr.index[0]: prior0}
  )


  # We said we'd keep track of the sum of the log of the probability
  # of the data for maximum likelihood calculation.
  log_likelihood = 0.0

  # (5) Iteratively apply Bayes' rule
  for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):
    # (5a) Calculate the new prior
    current_prior = process_matrix @ posteriors[previous_day]

    # (5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
    numerator = likelihoods[current_day] * current_prior

    # (5c) Calcluate the denominator of Bayes' Rule P(k)
    denominator = np.sum(numerator)

    # Execute full Bayes' Rule
    posteriors[current_day] = numerator / denominator

    # Add to the running sum of log likelihoods
    posteriors[current_day] = numerator / denominator

  return posteriors, log_likelihood


# Note that we're fixing sigma to a value just for the example
posteriors, log_likelihood = get_posteriors(smoothed, sigma=2.3)

hdis = highest_density_interval(posteriors, p=.9)

most_likely = posteriors.idxmax().rename('ML')

# Look into why you shift -1
result = pd.concat([most_likely, hdis], axis=1)

midd = result.tail(1).iloc[0]['ML']
low = result.tail(1).iloc[0]['Low_90']
high = result.tail(1).iloc[0]['High_90']

print(low)
print(midd)
print(high)
print(result.tail(1))
print(type(result))

def newPrediction(low, mid, high):
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="test"
    )

    sql = 'SELECT `total` FROM (SELECT `id`, `total` FROM `home__positive_cases` ORDER BY id DESC LIMIT 1) sub ORDER BY id ASC'

    mycursor = mydb.cursor()

    mycursor.execute(sql)

    lastDay = mycursor.fetchall()

    date = datetime.date(datetime.now())
    counter = 1
    sql2 = "INSERT INTO `future`(`date`, `low`, `mid`, `high`) VALUES (%s, %s, %s, %s)"
    val = (date + timedelta(days=5), int(lastDay[0][0] * low), int(lastDay[0][0] * mid), int(lastDay[0][0] * high))
    mycursor.execute(sql2, val)

    mydb.commit()


def newPredictionInit(low, mid, high):
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="test"
    )

    sql = 'SELECT `total` FROM (SELECT `id`, `total` FROM `home__positive_cases` ORDER BY id DESC LIMIT 5) sub ORDER BY id ASC'

    mycursor = mydb.cursor()

    mycursor.execute(sql)

    result = mycursor.fetchall()
    print(result[0][0])
    lastFiveDays = []
    for x in result:
        # print(x[0])
        lastFiveDays.append(x[0])


    date = datetime.date(datetime.now())
    counter = 1

    for x in lastFiveDays:
        sql = "INSERT INTO `future`(`date`, `low`, `mid`, `high`) VALUES (%s, %s, %s, %s)"
        val = (date + timedelta(days=counter), x*low, x*mid, x*high)
        mycursor.execute(sql, val)
        mydb.commit()


#newPrediction(low, midd, high)
#newPredictionInit(low, midd, high)