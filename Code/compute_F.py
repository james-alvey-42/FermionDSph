import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, simps, odeint
from scipy.interpolate import interp1d
from glob import glob
import pandas as pd

from plotting import plot_dwarfs, plot_FG, plot_FM_JA, plot_FM_JR
from data import get_dwarf, create_inner_df, load_FG_JR, rescale_FG_JR, load_FG_JA, load_rho, load_vesc_JA, load_vesc_JR, save_FM_JA, save_FM_JR, load_FM_JA, load_FM_JR

def FM(rho, vesc):
	return 3 * rho / (4 * np.pi * vesc**3)

def get_FM_data_JA(dwarf):
	rhodata, vescdata = load_rho(dwarf), load_vesc_JA(dwarf)
	save_data = {}
	save_data['r'] = rhodata['r']
	save_data['mid'] = FM(rhodata['mid'], vescdata['mid'])
	save_data['1sl'] = FM(rhodata['1sl'], vescdata['1su'])
	save_data['1su'] = FM(rhodata['1su'], vescdata['1sl'])
	save_data['2sl'] = FM(rhodata['2sl'], vescdata['2su'])
	save_data['2su'] = FM(rhodata['2su'], vescdata['2sl'])
	return save_data

def get_FM_data_JR(dwarf):
	rhodata, vescdata = load_rho(dwarf), load_vesc_JR(dwarf)
	save_data = {}
	save_data['r'] = rhodata['r']
	save_data['mid'] = FM(rhodata['mid'], vescdata['mid'])
	save_data['1sl'] = FM(rhodata['1sl'], vescdata['1su'])
	save_data['1su'] = FM(rhodata['1su'], vescdata['1sl'])
	save_data['2sl'] = FM(rhodata['2sl'], vescdata['2su'])
	save_data['2su'] = FM(rhodata['2su'], vescdata['2sl'])
	return save_data

if __name__ == '__main__':
	directories = glob('Final_Data/*/')

	# Rescale gaussian coarse grained PSDs
	'''
	for directory in directories:
		dwarf = get_dwarf(directory)
		FG_data = load_FG_JR(dwarf)
		rescale_FG_JR(FG_data, dwarf)
	'''

	# Plot Gaussian coarse-grained PSDs
	'''
	plot_dwarfs(load_FG_JA, plot_FG)
	'''

	# Creat inner data for gaussian coarse graining
	'''
	FG_cols = ['dwarf', 'FG/Msols^3km^-3kpc^-3', '2sl', '1sl', '1su', '2su']
	df = create_inner_df(load_FG_JA, cols=FG_cols)
	print(df.head())
	df.to_csv('Summary_Data/FG_inner.csv', index=True)
	'''

	# Compute maximal coarse graining
	'''
	for directory in directories:
		dwarf = get_dwarf(directory)
		FM_data_JA = get_FM_data_JA(dwarf)
		save_FM_JA(FM_data_JA, dwarf)
		FM_data_JR = get_FM_data_JR(dwarf)
		save_FM_JR(FM_data_JR, dwarf)
	'''

	# Plot maximal coarse graining
	'''
	plot_dwarfs(load_FM_JA, plot_FM_JA)
	plot_dwarfs(load_FM_JR, plot_FM_JR)
	'''

	# Generate inner df
	'''
	FM_cols = ['dwarf', 'FM/Msols^3km^-3kpc^-3', '2sl', '1sl', '1su', '2su']
	df = create_inner_df(load_FM_JA, cols=FM_cols)
	print(df.head())
	df.to_csv('Summary_Data/FM_JA_inner.csv', index=True)
	df = create_inner_df(load_FM_JR, cols=FM_cols)
	print(df.head())
	df.to_csv('Summary_Data/FM_JR_inner.csv', index=True)
	'''