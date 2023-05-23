
"""
=======================================================================================================================
DEPENDENCIES
------------

Centralize all project dependencies here.

: zach wolpe
: zach.wolpe@medibio.com.au
: 11 May 2023

=======================================================================================================================
"""

# Probabilistic Programming
from numpyro.infer import HMC, MCMC, NUTS, SA, Predictive, log_likelihood
from numpyro.examples.datasets import UCBADMIT, load_dataset
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro.distributions as dist
import arviz as az
import numpyro


# JAX: numeric computation
from jax.scipy.special import expit
import jax.numpy as jnp
import jax

# Statisitical models
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
from sklearn.metrics import r2_score
import statsmodels.api as sm
from pygam import LinearGAM
import pygam

# Plotting
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# Environment configuration
import platform,socket,re,uuid,json,psutil,logging
from logging.handlers import SysLogHandler
from colorlog import ColoredFormatter
import logging, platform, coloredlogs
from abc import ABC, abstractmethod
from socket import gethostname
from functools import wraps
import coloredlogs, logging
import warnings
import argparse
import socket
import sys
import os

# Numeric Analysis
import pandas as pd
import numpy as np
import random
import time
import math


warnings.filterwarnings('ignore')