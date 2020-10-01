import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, simps, odeint
from scipy.interpolate import interp1d
from glob import glob
import pandas as pd

from plotting import plot_dwarfs, plot_vesc, plot_vesc_JR
from data import get_dwarf, create_inner_df, load_M, load_M_JA, load_vesc_JA, save_vesc, load_vesc_JR, save_vesc_JR, create_inner_df

def get_vesc(r_arr, mass_arr):
	GN = 6.67430 * 1e-11 # m^3 kg^-1 s^-2
	Msol = 1.98847 * 1e30 # kg
	kpc = 3.086 * 1e19 # m
	GNnew = GN * kpc**(-3) * Msol # kpc^3 Msol^-1 s^-2
	vesc_arr = []
	for idx in range(0, len(r_arr)):
		rcut = r_arr[idx]
		vescsq = 2 * GNnew * mass_arr[-1] * np.power(r_arr[-1], -1.0) + 2 * GNnew * simps(mass_arr * np.power(r_arr, -2.0) * np.heaviside(r_arr - rcut, 1.0), x=r_arr)
		vesc = kpc * np.power(vescsq, 0.5) / 1e3
		vesc_arr.append(vesc)
	return np.array(vesc_arr)

def get_vesc_data(massdata):
	vescdata = {}
	vescdata['r'] = massdata['r']
	for label in ['mid', '1sl', '1su', '2sl', '2su']:
		vesc_arr = get_vesc(massdata['r'], massdata[label])
		vescdata[label] = vesc_arr
	return vescdata

if __name__ == '__main__':
	directories = glob('Final_Data/*/')

	# Generate vesc data from JA mass profile
	'''
	for directory in directories:
		dwarf = get_dwarf(directory)
		print(dwarf, '\n')
		mass_data = load_M_JA(dwarf)
		vesc_data = get_vesc_data(mass_data)
		save_vesc(vesc_data, dwarf)
	'''

	# Generate vesc data from JR mass profile
	'''
	for directory in directories:
		dwarf = get_dwarf(directory)
		print(dwarf, '\n')
		mass_data = load_M(dwarf)
		vesc_data = get_vesc_data(mass_data)
		save_vesc_JR(vesc_data, dwarf)
	'''
	

	# Plot vesc data
	'''
	plot_dwarfs(load_vesc_JA, plot_vesc)
	plot_dwarfs(load_vesc_JR, plot_vesc_JR)
	'''

	# Save inner values
	'''
	vesc_cols = ['dwarf', 'vesc/kms^-1', '2sl', '1sl', '1su', '2su']
	df_JA = create_inner_df(load_vesc_JA, cols=vesc_cols)
	df_JA.to_csv('Summary_Data/vesc_JA_inner.csv', index=True)
	df_JR = create_inner_df(load_vesc_JR, cols=vesc_cols)
	df_JR.to_csv('Summary_Data/vesc_JR_inner.csv', index=True)
	'''

