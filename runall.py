#!/usr/bin/env python3
import os
import sys
from sys import stdout
import argparse

parser = argparse.ArgumentParser(description="Run all 8 scenarios sequantially.")
parser.add_argument('simulations', type=int, default=100, help="Number of simulations to run")
args = parser.parse_args()
n_sims = args.simulations
print(n_sims)

print("Starting simulation {} times".format(n_sims))

print("Starting high intensity, high slope, high revenue gap")
os.system("python3 simulate.py high high high --simulations {}".format(n_sims))

print("Starting high intensity, high slope, low revenue gap")
os.system("python3 simulate.py high high low --simulations {}".format(n_sims))

print("Starting high intensity, low slope, high revenue gap")
os.system("python3 simulate.py high low high --simulations {}".format(n_sims))

print("Starting high intensity, low slope, low revenue gap")
os.system("python3 simulate.py high low low --simulations {}".format(n_sims))

print("Starting low intensity, high slope, high revenue gap")
os.system("python3 simulate.py low high high --simulations {}".format(n_sims))

print("Starting low intensity, high slope, low revenue gap")
os.system("python3 simulate.py low high low --simulations {}".format(n_sims))

print("Starting low intensity, low slope, high revenue gap")
os.system("python3 simulate.py low low high --simulations {}".format(n_sims))

print("Starting low intensity, low slope, low revenue gap")
os.system("python3 simulate.py low low low --simulations {}".format(n_sims))

