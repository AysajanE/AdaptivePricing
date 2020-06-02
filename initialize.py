#!/usr/bin/env python3
# Initialize adaptive pricing algorithm by generating initial nested protection levels, 
# each bucket representative revenues, and discount ratios for a given initial rate, 
# capacity, and demand intensity
def initialize(capacity, intensity, rates_init): 
    import itertools
    from operator import itemgetter
    import numpy as np
    from cvxopt import matrix, solvers

    # Parameter values
    n_class = 2
    los = 3
    combs = n_class * 7 * los

    # Calculate averate rates for each arrival day of week and los combination
    rates_arrival_los = [[rates_init[i, j],
                          rates_init[i, j] + rates_init[i, (j+1)%7],
                          rates_init[i, j] + rates_init[i, (j+1)%7] + rates_init[i, (j+2)%7]] 
                          for i, j in itertools.product(range(n_class), range(7))]
    # Store it as a numpy array
    rates_arrival_los = np.array(rates_arrival_los).reshape(n_class, 7, los)

    # Coefficients of objective function for LP
    obj_coefs = (-1) * rates_arrival_los.reshape(n_class * 7 * los)

    # Inequality equations, LHS
    # We have total number of 42 decision veriables, corresponding to total number of
    # rate class, arrival day of week and los combinations.
    # Column indexes 0-20 are associated with decision variables for rate class 1
    # Column indexes 21-41 are associated with decision variables for rate class 2
    G = np.zeros(7 * los * n_class * 7).reshape(7, n_class*7*los)
    # Arrivals that span Sunday stay night for rate class 1
    G[0,:(7*los)] = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # Arrivals that span Monday stay night for rate class 1
    G[1,:(7*los)] = [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    G[2,:(7*los)] = [0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    G[3,:(7*los)] = [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    G[4,:(7*los)] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    G[5,:(7*los)] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0]
    G[6,:(7*los)] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1]
    # Arrivals that span Sunday stay night for rate class 2
    G[0,(7*los):] = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # Arrivals that span Monday stay night for rate class 2
    G[1,(7*los):] = [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    G[2,(7*los):] = [0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    G[3,(7*los):] = [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    G[4,(7*los):] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    G[5,(7*los):] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0]
    G[6,(7*los):] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1]

    # identity matrix for expected demand constraints, G for capacity constraints,
    # Negative identity matrix for non-negativity
    G = np.concatenate((np.identity(combs), G, -np.identity(combs)), axis=0)
    # Inequality equations, RHS
    # For each rate class, number of arrivals for a stay night in question is half of
    # expected demand, which is capacity * intensity, then this expected demand is equally 
    # split between 6 arrival day, los combination that spans the stay night in question
    expDemand_each = (capacity * intensity * 0.5) / 6
    h = np.round(expDemand_each, decimals=0) * np.ones(n_class * 7 * los)
    # First h for expected demand, second component for capacity rhs
    # Third component for non-negativity rhs.
    h = np.concatenate((h, capacity * np.ones(7), np.zeros(combs)), axis=0)


    # Convert numpy arrays to cvxopt matrix forms
    c = matrix(obj_coefs, tc='d')
    G = matrix(G)
    h = matrix(h)

    # Solve LP, method = "interior point method" by default
    # Results are used for initialization purpose (warm start), so it should
    # not affect final algorithm performance after an efficient number of runs
    
    solvers.options['show_progress'] = False
    sol = solvers.lp(c, G, h)

    # Optimal solutions serve as booking limits: number of booking requests to accept for 
    # a given rate class, arrival day, los combination
    bkLimits = np.array(sol['x']).reshape(n_class, 7, los)
    bkLimits = np.round(bkLimits, decimals=0)

    # Dual values associated with demand constraints
    # Represent marginal contribution for the stay night revenue
    duals = np.array(sol['z'])[:(n_class*7*los)].reshape(n_class, 7, los)
    duals = np.round(duals, decimals=0)

    # tuple (0, 1, 2) represents rate class 1, Monday arrival and 3-night stay
    sun_stay_index = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 6, 1), (0, 6, 2), (0, 5, 2), 
                (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 6, 1), (1, 6, 2), (1, 5, 2)]

    mon_stay_index = [(0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 0, 1), (0, 0, 2), (0, 6, 2), 
                (1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 0, 1), (1, 0, 2), (1, 6, 2)]

    tue_stay_index = [(0, 2, 0), (0, 2, 1), (0, 2, 2), (0, 1, 1), (0, 1, 2), (0, 0, 2), 
                (1, 2, 0), (1, 2, 1), (1, 2, 2), (1, 1, 1), (1, 1, 2), (1, 0, 2)]

    wed_stay_index = [(0, 3, 0), (0, 3, 1), (0, 3, 2), (0, 2, 1), (0, 2, 2), (0, 1, 2), 
                (1, 3, 0), (1, 3, 1), (1, 3, 2), (1, 2, 1), (1, 2, 2), (1, 1, 2)]

    thr_stay_index = [(0, 4, 0), (0, 4, 1), (0, 4, 2), (0, 3, 1), (0, 3, 2), (0, 2, 2), 
                (1, 4, 0), (1, 4, 1), (1, 4, 2), (1, 3, 1), (1, 3, 2), (1, 2, 2)]

    fri_stay_index = [(0, 5, 0), (0, 5, 1), (0, 5, 2), (0, 4, 1), (0, 4, 2), (0, 3, 2), 
                (1, 5, 0), (1, 5, 1), (1, 5, 2), (1, 4, 1), (1, 4, 2), (1, 3, 2)]

    sat_stay_index = [(0, 6, 0), (0, 6, 1), (0, 6, 2), (0, 5, 1), (0, 5, 2), (0, 4, 2), 
                (1, 6, 0), (1, 6, 1), (1, 6, 2), (1, 5, 1), (1, 5, 2), (1, 4, 2)]

    # Create one complete index list
    stay_index = [sun_stay_index, mon_stay_index, tue_stay_index, wed_stay_index,
                 thr_stay_index, fri_stay_index, sat_stay_index]

    # Create virtual buckets with booking classes in them
    # Use a copy of the index list in order to keep the index list unchanged for later use
    stay_index_cp = stay_index.copy()
    max_buckets = 5
    buckets_all = []

    for index_ls in stay_index_cp:
        buckets = [[] for i in range(max_buckets)]
        duals_ls = [duals[item] for item in index_ls]
        n_buckets = 1

        # Stop clustering when the number of buckets is higher than max_buckets of 5 or
        # when the index list is empty
        while n_buckets <= max_buckets and index_ls:
            duals_left = [duals[item] for item in index_ls]
            duals_max = np.max(duals_left)
            # Keep track of the elements to remove from the index list in 
            # each bucket creation iteration
            item_delete_index = []
            # Create lower and upper limits for buckets
            lower = ((max_buckets-n_buckets) / (max_buckets-n_buckets+1)) * duals_max
            upper = duals_max
            # Cluster each item into appropriate buckets
            for item in index_ls:
                item_index = index_ls.index(item)
                if duals[item] >= lower: 
                    buckets[n_buckets-1].append(item)
                    item_delete_index.append(item_index)
            # Update index list for the next iteration by removing elements that have been
            # already clustered
            index_ls = [item for item in index_ls if index_ls.index(item) not in item_delete_index]
            # Update bucket number
            n_buckets += 1
            
        buckets_all.append(buckets)
        

    # Filter out empty lists where number of buckets for a stay night is less than 5
    buckets_all = [list(filter(None, elem)) for elem in buckets_all]


    # Number of buckets in each stay night
    n_buckets = list(map(len, buckets_all))

    # Calculate protection levels for each bucket in each stay night of the week
    # Store partitioned protection levels for all days of a week
    ptLevels_ptd = []
    for buckets_each in buckets_all:
        # Store partitioned protection levels for each stay night
        levels = []
        # Partitioned protection levels
        for bucket in buckets_each:
            # If there is only one element in the bucket, then it is not iterable
            # and code will throw out TypeError. Therefore, we implement try...except.
            try:
                level = list(itemgetter(*bucket)(bkLimits))
                level = np.sum(np.array(level).reshape(len(level)))
            except TypeError:
                level = bkLimits[bucket[0]]
            levels.append(level)
        ptLevels_ptd.append(levels)


    # Compute nested protection levels
    ptLevels = [np.cumsum(ptLevels_ptd[i]) for i in range(7)]

    # Store number of rda combinations in a bucket for a stay night
    bkt_length = [list(map(len, buckets_all[i])) for i in range(7)]

    # Calculate average revenue for each bucket in each stay night of the week
    # Store calculated average revenues for all days of a week
    revenues_avg = []
    for buckets_each in buckets_all:
        # Store average revenue for each stay night
        revenue_each = []
        for bucket in buckets_each:
            # If there is only one element in the bucket, then it is not iterable
            # and code will throw out TypeError. Therefore, we implement try...except.
            try:
                revenue = list(itemgetter(*bucket)(rates_arrival_los))
                revenue = np.mean(np.array(revenue).reshape(len(revenue)))
            except TypeError:
                revenue = rates_arrival_los[bucket[0]]
            revenue_each.append(revenue)
        revenues_avg.append(revenue_each)


    # Calculate minimum dual values for each bucket in each stay night of the week
    # Store output for all days of a week
    duals_min = []
    for buckets_each in buckets_all:
        # Store average revenue for each stay night
        dual_vals = []
        for bucket in buckets_each:
            # If there is only one element in the bucket, then it is not iterable
            # and code will throw out TypeError. Therefore, we implement try...except.
            try:
                dual_val = list(itemgetter(*bucket)(duals))
                dual_val = np.min(np.array(dual_val).reshape(len(dual_val)))
            except TypeError:
                dual_val = duals[bucket[0]]
            dual_vals.append(dual_val)
        duals_min.append(dual_vals)


    # Calculate dual values difference and find the average difference
    # Here, implement Algorithm 2 from the paper
    duals_diff = []
    for vals in duals_min:
        duals_diff_each = []
        [duals_diff_each.append(vals[i] - vals[i+1]) for i in range(len(vals) - 1)]
        duals_diff.append(duals_diff_each)

    # After calculation the difference between two adjacent dual values, we calculate
    # the average dual difference by diving the difference by the number of booking classes
    # in that bucket
    avg_diff = []
    for diff, length in zip(duals_diff, bkt_length):
        avg_diff.append(np.array(diff) / np.array(length[:-1]))

    # Calculate final representative rates for each bucket in each stay night
    # by taking the maximum values of three quantities
    rates = []
    for i in range(7):
        rates_each = []
        # This is for the last bucket in each stay night
        maxVal = max(revenues_avg[i][n_buckets[i]-1], duals_min[i][n_buckets[i]-1])
        rates_each.append(maxVal)
        # This is for all the remaining buckets, counting backwards
        for j in range(n_buckets[i]-2, -1, -1):
            revenues_adj = rates_each[0] + avg_diff[i][j]
            final_rate = max(revenues_avg[i][j], duals_min[i][j], revenues_adj)
            # Since counting backward, we insert each rate at the beginning of the list
            rates_each.insert(0, final_rate)
        rates.append(rates_each)
        rates = [[round(rate, 0) for rate in rates[i]] for i in range(7)]


    # Discount ratio
    ratios = []
    for i in range(7):
        ratios_each = []
        for j in range(n_buckets[i]):
            try:
                ratio = rates[i][j] / rates[i][0]
            except ZeroDivisionError:
                ratio = 0.5
            ratios_each.append(ratio)
        ratios.append(ratios_each)
        ratios = [[round(ratio, 4) for ratio in ratios[i]] for i in range(7)]
    
    # Return protection levels, representative rates, and ratios for implementation in the next step.
    return(ptLevels_ptd, ptLevels, rates, ratios)


