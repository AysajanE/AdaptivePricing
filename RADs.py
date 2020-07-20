#!/usr/bin/env python3

# This script lists all the booking classes that cover a certain 
# stay night, from Sunday to Saturday.

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
# Create one complete index list
stay_index = [sun_stay_index, mon_stay_index, tue_stay_index, wed_stay_index,
        thr_stay_index, fri_stay_index, sat_stay_index]


sun_stay_index_wk1 = [(0, 0, 0), (0, 0, 1), (0, 0, 2),  
            (1, 0, 0), (1, 0, 1), (1, 0, 2) ]

mon_stay_index_wk1 = [(0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 0, 1), (0, 0, 2),  
            (1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 0, 1), (1, 0, 2)]
stay_index_wk1 =[sun_stay_index_wk1, mon_stay_index_wk1, tue_stay_index, 
        wed_stay_index, thr_stay_index, fri_stay_index, sat_stay_index] 
