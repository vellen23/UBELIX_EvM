import os
import numpy as np
import sys
import sklearn
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from tkinter import *
from sklearn.decomposition import NMF
import pandas as pd
import random

def clusters2connection(con_trial, clusters):
    