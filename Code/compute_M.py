import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, simps, odeint
from scipy.interpolate import interp1d
from glob import glob
import pandas as pd

from plotting import plot_rho, plot_dwarfs, plot_mass, plot_mass_JR
from data import get_dwarf, create_inner_df, load_rho, load_M, load_M_JA, save_M

def get_mass_data(rhodata):
    massdata = {}
    massdata['r'] = rhodata['r']
    for label in ['mid', '1sl', '1su', '2sl', '2su']:
        rho = rhodata[label]
        rhofn = interp1d(rhodata['r'], rho, kind='linear', fill_value='extrapolate')
        
        def dmdr(m, r):
            dmassdr = 4 * np.pi * np.power(r, 2) * rhofn(r)
            return dmassdr

        m0 = (4/3) * np.pi * rhodata['r'][0]**3 * rho[0]
        m = odeint(dmdr, m0, rhodata['r'])
        massdata[label] = m.T[0]
    return massdata

if __name__ == '__main__':
	directories = glob('Final_Data/*/')
	
	# Generate inner data for rho
	'''
	rho_cols = ['dwarf', 'rho/Msolkpc^-3', '2sl', '1sl', '1su', '2su']
	rho_df = create_inner_df(load_rho, rho_cols)
	print(rho_df.head())
	rho_df.to_csv('Summary_Data/rho_inner.csv', index=True)
	'''
	
	# Plot all rho profiles
	'''
	plot_dwarfs(load_rho, plot_rho)
	'''

	# Solve ODE to find the mass profile
	'''
	for directory in directories:
		dwarf = get_dwarf(directory)
		rho_data = load_rho(dwarf)
		mass_data = get_mass_data(rho_data)
		save_M(mass_data, dwarf)
	'''

	# Plot mass profile and compare to JR data
	'''
	plot_dwarfs(load_M, plot_mass_JR)
	plot_dwarfs(load_M_JA, plot_mass)
	'''



