import numpy as np
import pandas as pd
from glob import glob
from scipy.interpolate import interp1d

def get_dwarf(directory):
	dwarf = directory.split('/')[-2]
	return dwarf

def get_inner(dwarf, data):
	return np.array([dwarf, data['mid'][0], data['2sl'][0], data['1sl'][0], data['1su'][0], data['2su'][0]])

def create_inner_df(load_fn, cols):
	directories = glob('Final_Data/*/')
	df = pd.DataFrame(columns=cols)
	for directory in directories:
		dwarf = directory.split('/')[-2]
		data = load_fn(dwarf)
		inner_values = get_inner(dwarf, data)
		df = df.append(pd.DataFrame([inner_values], columns=cols))
	df = df.set_index('dwarf')
	return df

def load_rho(dwarf):
	data = np.loadtxt('Final_Data/' + dwarf + '/output_rho.txt', unpack=False)
	return {'r': data[:, 0], 'mid': data[:, 1], 
			'1sl': data[:, 2], '1su':data[:, 3],
			'2sl': data[:, 4], '2su': data[:, 5]}

def load_M(dwarf):
	data = np.loadtxt('Final_Data/' + dwarf + '/output_M.txt', unpack=False)
	return {'r': data[:, 0], 'mid': data[:, 1], 
			'1sl': data[:, 2], '1su':data[:, 3],
			'2sl': data[:, 4], '2su': data[:, 5]}

def load_M_JA(dwarf):
	data = np.loadtxt('Final_Data/' + dwarf + '/output_M_JA.txt', unpack=False)
	return {'r': data[:, 0], 'mid': data[:, 1], 
			'1sl': data[:, 2], '1su':data[:, 3],
			'2sl': data[:, 4], '2su': data[:, 5]}

def load_vesc_JA(dwarf):
	data = np.loadtxt('Final_Data/' + dwarf + '/output_vesc_JA.txt', unpack=False)
	return {'r': data[:, 0], 'mid': data[:, 1], 
			'1sl': data[:, 2], '1su':data[:, 3],
			'2sl': data[:, 4], '2su': data[:, 5]}

def load_vesc_JR(dwarf):
	data = np.loadtxt('Final_Data/' + dwarf + '/output_vesc_JR.txt', unpack=False)
	return {'r': data[:, 0], 'mid': data[:, 1], 
			'1sl': data[:, 2], '1su':data[:, 3],
			'2sl': data[:, 4], '2su': data[:, 5]}

def load_FG_JR(dwarf):
	data = np.loadtxt('Final_Data/' + dwarf + '/output_F_DF.txt', unpack=False)
	return {'r': data[:, 0], 'mid': data[:, 1], 
			'1sl': data[:, 2], '1su':data[:, 3],
			'2sl': data[:, 4], '2su': data[:, 5]}

def load_FG_JA(dwarf):
	data = np.loadtxt('Final_Data/' + dwarf + '/output_F_DF_JA.txt', unpack=False)
	return {'r': data[:, 0], 'mid': data[:, 1], 
			'1sl': data[:, 2], '1su':data[:, 3],
			'2sl': data[:, 4], '2su': data[:, 5]}

def load_FM_JA(dwarf):
	data = np.loadtxt('Final_Data/' + dwarf + '/output_FM_JA.txt', unpack=False)
	return {'r': data[:, 0], 'mid': data[:, 1], 
			'1sl': data[:, 2], '1su':data[:, 3],
			'2sl': data[:, 4], '2su': data[:, 5]}

def load_FM_JR(dwarf):
	data = np.loadtxt('Final_Data/' + dwarf + '/output_FM_JR.txt', unpack=False)
	return {'r': data[:, 0], 'mid': data[:, 1], 
			'1sl': data[:, 2], '1su':data[:, 3],
			'2sl': data[:, 4], '2su': data[:, 5]}

def load_summary_data(file):
	df = pd.read_csv('Summary_Data/' + file)
	df = df.set_index('dwarf')
	return df

def rescale_FG_JR(data, dwarf):
	prefactor = ((4 * np.pi)/3) * (1/(2* np.pi)**(3/2))
	save_arr = np.empty((len(data['r']), 6))
	save_arr[:, 0] = data['r']
	save_arr[:, 1] = data['mid'] * prefactor
	save_arr[:, 2] = data['1sl'] * prefactor
	save_arr[:, 3] = data['1su'] * prefactor
	save_arr[:, 4] = data['2sl'] * prefactor
	save_arr[:, 5] = data['2su'] * prefactor
	np.savetxt('Final_data/{}/output_F_DF_JA.txt'.format(dwarf), save_arr)

def save_M(data, dwarf):
	save_arr = np.empty((len(data['r']), 6))
	save_arr[:, 0] = data['r']
	save_arr[:, 1] = data['mid']
	save_arr[:, 2] = data['1sl']
	save_arr[:, 3] = data['1su']
	save_arr[:, 4] = data['2sl']
	save_arr[:, 5] = data['2su']
	np.savetxt('Final_Data/{}/output_M_JA.txt'.format(dwarf), save_arr)

def save_vesc(data, dwarf):
	save_arr = np.empty((len(data['r']), 6))
	save_arr[:, 0] = data['r']
	save_arr[:, 1] = data['mid']
	save_arr[:, 2] = data['1sl']
	save_arr[:, 3] = data['1su']
	save_arr[:, 4] = data['2sl']
	save_arr[:, 5] = data['2su']
	np.savetxt('Final_Data/{}/output_vesc_JA.txt'.format(dwarf), save_arr)

def save_vesc_JR(data, dwarf):
	save_arr = np.empty((len(data['r']), 6))
	save_arr[:, 0] = data['r']
	save_arr[:, 1] = data['mid']
	save_arr[:, 2] = data['1sl']
	save_arr[:, 3] = data['1su']
	save_arr[:, 4] = data['2sl']
	save_arr[:, 5] = data['2su']
	np.savetxt('Final_Data/{}/output_vesc_JR.txt'.format(dwarf), save_arr)

def save_FM_JA(data, dwarf):
	save_arr = np.empty((len(data['r']), 6))
	save_arr[:, 0] = data['r']
	save_arr[:, 1] = data['mid']
	save_arr[:, 2] = data['1sl']
	save_arr[:, 3] = data['1su']
	save_arr[:, 4] = data['2sl']
	save_arr[:, 5] = data['2su']
	np.savetxt('Final_Data/{}/output_FM_JA.txt'.format(dwarf), save_arr)

def save_FM_JR(data, dwarf):
	save_arr = np.empty((len(data['r']), 6))
	save_arr[:, 0] = data['r']
	save_arr[:, 1] = data['mid']
	save_arr[:, 2] = data['1sl']
	save_arr[:, 3] = data['1su']
	save_arr[:, 4] = data['2sl']
	save_arr[:, 5] = data['2su']
	np.savetxt('Final_Data/{}/output_FM_JR.txt'.format(dwarf), save_arr)

def load_read_prior():
	r, beta = np.loadtxt('Beta_Data/read_prior.txt', skiprows=0, unpack=True)
	return r, beta

def interpolate_simulation(filename):
	r, _, _, beta = np.loadtxt(filename, unpack=True)
	return interp1d(r, beta, kind='linear')

def get_mass_mix(fdvalue, RPfile):
	mass, mix = np.loadtxt(RPfile, unpack=True)
	mix_fn = interp1d(mass, mix, kind='cubic', fill_value='extrapolate')
	masses = np.geomspace(fdvalue, max(mass), 1000)
	mix_arr = mix_fn(masses)
	masses = np.append(fdvalue, masses)
	mass_add = masses[-1]
	masses = np.append(masses, mass_add)
	mix_arr = np.append(1e-6, mix_arr)
	mix_arr = np.append(mix_arr, 1e-13)
	return masses, mix_arr

def get_mass_mix_nointerp(fdvalue, RPfile):
	mass, mix = np.loadtxt(RPfile, unpack=True)
	mass_ret, mix_ret = np.empty(len(mass) + 3), np.empty(len(mix) + 3)
	mass_ret[2:-1], mix_ret[2:-1] = mass, mix
	mass_ret[0], mix_ret[0] = fdvalue, 1e-7
	mass_ret[1], mix_ret[1] = fdvalue, mix[0]
	mass_ret[-1], mix_ret[-1] = mass[-1], 1e-13
	return mass_ret, mix_ret
