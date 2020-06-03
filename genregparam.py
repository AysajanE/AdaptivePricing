#!/usr/bin/env python3
# Initiate linear demand curve parameters for simulation analysis
# One linear demand curve is initiated for each rate class, arrival day of week and 
# los combination

import itertools
import numpy as np
from scipy.stats import norm
from scipy.stats import poisson
from sklearn.linear_model import LinearRegression

# Consider two rate classes for simplicity
n_class = 2
# Consider Length-of-Stay (los) of upto three days
los = 3

def linparams(capacity, intensity, slope, rates_init):

    # Take the slope of -0.1 for rate class 1, then the mean true demand point for any arrival date/lengthof-
    # stay combination is passed through by the line defined by this slope; 30 points on the price
    # axis for each arrival date/length-of-stay combination over the first 21 simulation days
    # are randomly generated from the interval (init_rate-50, init_rate+50)
    # Sunday-one night stay as an example
    drawsize = 30
    # Random draw for one night stay for each day of week
    randomRates_one = [np.random.randint(rates_init[i, j]-50, rates_init[i, j]+50+1, drawsize)
                         for i, j in itertools.product(range(n_class), range(7))]
    randomRates_one = np.array(randomRates_one).reshape(n_class, 7, drawsize)
    # Construct rates for more than one night stay, up to 3-night stay
    # Sunday night arrival, two night stay rates equal Sunday one night + Monday one night stay
    randomRates_two = [randomRates_one[i, j%7]
                         + randomRates_one[i, (j+1)%7]
                         for i, j in itertools.product(range(n_class), range(7))]

    randomRates_three = [randomRates_one[i, j%7]
                           + randomRates_one[i, (j+1)%7]
                           + randomRates_one[i, (j+2)%7]
                         for i, j in itertools.product(range(n_class), range(7))]

    # For each arrival date, los can be 1-night, 2-night, or 3-night.
    # Accordingly, the rates will be corresponding to different los.
    # e.g., for Mon arrival, 2-night stay, the rate equals Mon single night rate + Tu single night rate
    randomRates_one = np.array(randomRates_one).reshape(n_class, 7, drawsize)
    randomRates_two = np.array(randomRates_two).reshape(n_class, 7, drawsize)
    randomRates_three = np.array(randomRates_three).reshape(n_class, 7, drawsize)

    randomRates = [(randomRates_one[i, j], randomRates_two[i, j], randomRates_three[i, j])
                   for i, j in itertools.product(range(n_class), range(7))]
    # axis 0 = n_class, axis 1 = day of week, axis 2 = los, and axis 3 = random draws
    randomRates = np.array(randomRates).reshape(n_class, 7, los, drawsize)

    # Calculate averate rates for each arrival day of week and los combination
    rates_arrival_los = [[rates_init[i, j],
                          rates_init[i, j] + rates_init[i, (j+1)%7],
                          rates_init[i, j] + rates_init[i, (j+1)%7] + rates_init[i, (j+2)%7]]
                          for i, j in itertools.product(range(n_class), range(7))]
    # Store it as a numpy array
    rates_arrival_los = np.array(rates_arrival_los).reshape(n_class, 7, los)


    # Calculate y-intercepts, assuming los distribution for a stay night is 1/3, 1/3, and 1/3
    # There exist one 1-night stay, two 2-night stays and three 3-night stays that cover
    # stay night in question.
    # Each rate class contribute half of the demand for each stay night
    half_demand = 0.5 * capacity * intensity
    # Intercepts for 1-night stay
    A1 = np.array([[1, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 1]])
    b1 = [(half_demand * 1/3 - slope[i] * rates_arrival_los[i, j, 0])
                   for i, j in itertools.product(range(n_class), range(7))]
    b1 = np.array(b1).reshape(n_class, 7)
    x1 = [np.linalg.solve(A1, b1[i]) for i in range(n_class)]

    # Intercepts for 2-night stay
    A2 = np.array([[1, 0, 0, 0, 0, 0, 1],
                   [1, 1, 0, 0, 0, 0, 0],
                   [0, 1, 1, 0, 0, 0, 0],
                   [0, 0, 1, 1, 0, 0, 0],
                   [0, 0, 0, 1, 1, 0, 0],
                   [0, 0, 0, 0, 1, 1, 0],
                   [0, 0, 0, 0, 0, 1, 1]])
    b2 = [(half_demand * 1/3 - slope[i] * (rates_arrival_los[i, j, 1]
                                                   +rates_arrival_los[i, (j-1+7)%7, 1]))
                   for i, j in itertools.product(range(n_class), range(7))]
    b2 = np.array(b2).reshape(n_class, 7)
    x2 = [np.linalg.solve(A2, b2[i]) for i in range(n_class)]

    # Intercepts for 3-night stay
    A3 = np.array([[1, 0, 0, 0, 0, 1, 1],
                   [1, 1, 0, 0, 0, 0, 1],
                   [1, 1, 1, 0, 0, 0, 0],
                   [0, 1, 1, 1, 0, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0],
                   [0, 0, 0, 1, 1, 1, 0],
                   [0, 0, 0, 0, 1, 1, 1]])
    b3 = [(half_demand * 1/3 - slope[i] * (rates_arrival_los[i, j, 2]
                                                   +rates_arrival_los[i, (j-1+7)%7, 2]
                                                   +rates_arrival_los[i, (j-2+7)%7, 2]))
                 for i, j in itertools.product(range(n_class), range(7))]
    b3 = np.array(b3).reshape(n_class, 7)
    x3 = [np.linalg.solve(A3, b3[i]) for i in range(n_class)]

    # Create intercepts matrix for all arrival day-los combinations
    # Total 2 * 7 * 3 = 42 intercepts
    intercepts = np.concatenate((np.array(x1), np.array(x2), np.array(x3)), axis=1)
    intercepts = np.reshape(intercepts, (n_class, los, 7), order='C')
    intercepts = np.array([np.transpose(intercepts[i]) for i in range(n_class)])

    # Mean of Poisson demand that is generated by the slope and y-intercept
    # One mean demand for each of the 30 randomly drawn price
    meanDemand = [intercepts[i,j, k] + slope[i] * randomRates[i, j, k, l]
                    for i, j, k, l in itertools.product(range(n_class), range(7), range(los), range(drawsize))]
    meanDemand = np.array(meanDemand).reshape(n_class, 7, los, drawsize)
    zeros = np.zeros(n_class * 7 * los * drawsize).reshape(n_class, 7, los, drawsize)
    # Can't have megative demand, so truncate mean demand at zero
    meanDemand = np.maximum(meanDemand, zeros)

    # Generate random demand for each day of week and length of stay combination
    randomDemand = [poisson.rvs(mu, size=1) for mu in np.nditer(meanDemand)]
    randomDemand = np.array(randomDemand).reshape(n_class, 7, los, drawsize)


    # Create linear regression object
    slopes_update = []
    intercepts_update = []
    regr = LinearRegression()
    for i, j, k in itertools.product(range(n_class), range(7), range(los)):
        regr.fit(randomRates[i, j, k, :].reshape(drawsize, -1), randomDemand[i, j, k, :])
        intercepts_update.append(regr.intercept_)
        slopes_update.append(regr.coef_)

    # Store intercepts and slopes as numpy array
    slopes_update = np.array(slopes_update).reshape(n_class, 7, los)
    intercepts_update = np.array(intercepts_update).reshape(n_class, 7, los)

    slopes_update = np.round(slopes_update, 2)
    intercepts_update = np.round(intercepts_update, 0)

    return (slopes_update, intercepts_update)
