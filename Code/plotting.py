import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.interpolate import interp1d
from glob import glob
from data import load_read_prior, interpolate_simulation

def plot_rho(data, dwarf):
	xlabel, ylabel = r'$r\,\mathrm{[kpc]}$', r'$\rho(r)\,\mathrm{[M}_{\odot} \, \mathrm{kpc}^{-3}\mathrm{]}$'

	color = '#994B92'

	fig = plt.figure(figsize=(5, 5))

	ax = plt.subplot(1, 1, 1)
	plt.sca(ax)

	xmin, xmax = min(data['r']), max(data['r'])

	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	
	plt.fill_between(data['r'], data['2sl'], data['2su'], 
		color=color, alpha=0.2, linewidth=0.0)
	plt.fill_between(data['r'], data['1sl'], data['1su'], 
		color=color, alpha=0.4, linewidth=0.0)
	plt.plot(data['r'], data['1sl'], c=color, lw=0.4, alpha=0.8)
	plt.plot(data['r'], data['1su'], c=color, lw=0.4, alpha=0.8)
	plt.plot(data['r'], data['2sl'], c=color, lw=0.4, alpha=0.3)
	plt.plot(data['r'], data['2su'], c=color, lw=0.4, alpha=0.3)
	plt.plot(data['r'], data['mid'], c=color, lw=1.8, label=dwarf)

	plt.xlim(xmin, xmax)
	
	leg = plt.legend(fontsize=18, handlelength=0, handletextpad=0, loc='lower left')
	for item in leg.legendHandles:
		item.set_visible(False)

	plt.savefig('Plots/rho_{}.pdf'.format(dwarf))
	plt.close(fig)

def plot_mass(data, dwarf):
	xlabel, ylabel = r'$r\,\mathrm{[kpc]}$', r'$M(r)\,\mathrm{[M}_{\odot}\mathrm{]}$'

	color = '#994B92'

	fig = plt.figure(figsize=(5, 5))

	ax = plt.subplot(1, 1, 1)
	plt.sca(ax)

	xmin, xmax = min(data['r']), max(data['r'])

	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	
	plt.fill_between(data['r'], data['2sl'], data['2su'], 
		color=color, alpha=0.2, linewidth=0.0)
	plt.fill_between(data['r'], data['1sl'], data['1su'], 
		color=color, alpha=0.4, linewidth=0.0)
	plt.plot(data['r'], data['1sl'], c=color, lw=0.4, alpha=0.8)
	plt.plot(data['r'], data['1su'], c=color, lw=0.4, alpha=0.8)
	plt.plot(data['r'], data['2sl'], c=color, lw=0.4, alpha=0.3)
	plt.plot(data['r'], data['2su'], c=color, lw=0.4, alpha=0.3)
	plt.plot(data['r'], data['mid'], c=color, lw=1.8, label=dwarf)

	plt.xlim(xmin, xmax)
	
	leg = plt.legend(fontsize=18, handlelength=0, handletextpad=0, loc='lower left')
	for item in leg.legendHandles:
		item.set_visible(False)

	plt.savefig('Plots/mass_{}.pdf'.format(dwarf))
	plt.close(fig)

def plot_mass_JR(data, dwarf):
	xlabel, ylabel = r'$r\,\mathrm{[kpc]}$', r'$M(r)\,\mathrm{[M}_{\odot}\mathrm{]}$'

	color = '#994B92'

	fig = plt.figure(figsize=(5, 5))

	ax = plt.subplot(1, 1, 1)
	plt.sca(ax)

	xmin, xmax = min(data['r']), max(data['r'])

	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	
	plt.fill_between(data['r'], data['2sl'], data['2su'], 
		color=color, alpha=0.2, linewidth=0.0)
	plt.fill_between(data['r'], data['1sl'], data['1su'], 
		color=color, alpha=0.4, linewidth=0.0)
	plt.plot(data['r'], data['1sl'], c=color, lw=0.4, alpha=0.8)
	plt.plot(data['r'], data['1su'], c=color, lw=0.4, alpha=0.8)
	plt.plot(data['r'], data['2sl'], c=color, lw=0.4, alpha=0.3)
	plt.plot(data['r'], data['2su'], c=color, lw=0.4, alpha=0.3)
	plt.plot(data['r'], data['mid'], c=color, lw=1.8, label=dwarf)

	plt.xlim(xmin, xmax)
	
	leg = plt.legend(fontsize=18, handlelength=0, handletextpad=0, loc='lower left')
	for item in leg.legendHandles:
		item.set_visible(False)

	plt.savefig('Plots/massJR_{}.pdf'.format(dwarf))
	plt.close(fig)

def plot_vesc(data, dwarf):
	xlabel, ylabel = r'$r\,\mathrm{[kpc]}$', r'$v_{\mathrm{esc}}(r)\,\mathrm{[km}\,\mathrm{s}^{-1}\mathrm{]}$'

	color = '#994B92'

	fig = plt.figure(figsize=(5, 5))

	ax = plt.subplot(1, 1, 1)
	plt.sca(ax)

	xmin, xmax = min(data['r']), max(data['r'])

	plt.xscale('log')
	plt.yscale('linear')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	
	plt.fill_between(data['r'], data['2sl'], data['2su'], 
		color=color, alpha=0.2, linewidth=0.0)
	plt.fill_between(data['r'], data['1sl'], data['1su'], 
		color=color, alpha=0.4, linewidth=0.0)
	plt.plot(data['r'], data['1sl'], c=color, lw=0.4, alpha=0.8)
	plt.plot(data['r'], data['1su'], c=color, lw=0.4, alpha=0.8)
	plt.plot(data['r'], data['2sl'], c=color, lw=0.4, alpha=0.3)
	plt.plot(data['r'], data['2su'], c=color, lw=0.4, alpha=0.3)
	plt.plot(data['r'], data['mid'], c=color, lw=1.8, label=dwarf)

	plt.xlim(xmin, xmax)
	
	leg = plt.legend(fontsize=18, handlelength=0, handletextpad=0, loc='lower left')
	for item in leg.legendHandles:
		item.set_visible(False)

	plt.savefig('Plots/vesc_{}.pdf'.format(dwarf))
	plt.close(fig)

def plot_vesc_JR(data, dwarf):
	xlabel, ylabel = r'$r\,\mathrm{[kpc]}$', r'$v_{\mathrm{esc}}(r)\,\mathrm{[km}\,\mathrm{s}^{-1}\mathrm{]}$'

	color = '#994B92'

	fig = plt.figure(figsize=(5, 5))

	ax = plt.subplot(1, 1, 1)
	plt.sca(ax)

	xmin, xmax = min(data['r']), max(data['r'])

	plt.xscale('log')
	plt.yscale('linear')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	
	plt.fill_between(data['r'], data['2sl'], data['2su'], 
		color=color, alpha=0.2, linewidth=0.0)
	plt.fill_between(data['r'], data['1sl'], data['1su'], 
		color=color, alpha=0.4, linewidth=0.0)
	plt.plot(data['r'], data['1sl'], c=color, lw=0.4, alpha=0.8)
	plt.plot(data['r'], data['1su'], c=color, lw=0.4, alpha=0.8)
	plt.plot(data['r'], data['2sl'], c=color, lw=0.4, alpha=0.3)
	plt.plot(data['r'], data['2su'], c=color, lw=0.4, alpha=0.3)
	plt.plot(data['r'], data['mid'], c=color, lw=1.8, label=dwarf)

	plt.xlim(xmin, xmax)
	
	leg = plt.legend(fontsize=18, handlelength=0, handletextpad=0, loc='lower left')
	for item in leg.legendHandles:
		item.set_visible(False)

	plt.savefig('Plots/vescJR_{}.pdf'.format(dwarf))
	plt.close(fig)

def plot_FG(data, dwarf):
	xlabel, ylabel = r'$r\,\mathrm{[kpc]}$', r'$F_G(r)\,\mathrm{[M}_\odot\,\mathrm{kpc}^{-3}\,\mathrm{km}^{-3}\,\mathrm{s}^{3}\mathrm{]}$'

	color = '#994B92'

	fig = plt.figure(figsize=(5, 5))

	ax = plt.subplot(1, 1, 1)
	plt.sca(ax)

	xmin, xmax = min(data['r']), max(data['r'])

	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	
	plt.fill_between(data['r'], data['2sl'], data['2su'], 
		color=color, alpha=0.2, linewidth=0.0)
	plt.fill_between(data['r'], data['1sl'], data['1su'], 
		color=color, alpha=0.4, linewidth=0.0)
	plt.plot(data['r'], data['1sl'], c=color, lw=0.4, alpha=0.8)
	plt.plot(data['r'], data['1su'], c=color, lw=0.4, alpha=0.8)
	plt.plot(data['r'], data['2sl'], c=color, lw=0.4, alpha=0.3)
	plt.plot(data['r'], data['2su'], c=color, lw=0.4, alpha=0.3)
	plt.plot(data['r'], data['mid'], c=color, lw=1.8, label=dwarf)

	plt.xlim(xmin, xmax)
	
	leg = plt.legend(fontsize=18, handlelength=0, handletextpad=0, loc='lower left')
	for item in leg.legendHandles:
		item.set_visible(False)

	plt.savefig('Plots/FG_{}.pdf'.format(dwarf))
	plt.close(fig)

def plot_FM_JA(data, dwarf):
	xlabel, ylabel = r'$r\,\mathrm{[kpc]}$', r'$F_M(r)\,\mathrm{[M}_\odot\,\mathrm{kpc}^{-3}\,\mathrm{km}^{-3}\,\mathrm{s}^{3}\mathrm{]}$'

	color = '#994B92'

	fig = plt.figure(figsize=(5, 5))

	ax = plt.subplot(1, 1, 1)
	plt.sca(ax)

	xmin, xmax = min(data['r']), max(data['r'])

	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	
	plt.fill_between(data['r'], data['2sl'], data['2su'], 
		color=color, alpha=0.2, linewidth=0.0)
	plt.fill_between(data['r'], data['1sl'], data['1su'], 
		color=color, alpha=0.4, linewidth=0.0)
	plt.plot(data['r'], data['1sl'], c=color, lw=0.4, alpha=0.8)
	plt.plot(data['r'], data['1su'], c=color, lw=0.4, alpha=0.8)
	plt.plot(data['r'], data['2sl'], c=color, lw=0.4, alpha=0.3)
	plt.plot(data['r'], data['2su'], c=color, lw=0.4, alpha=0.3)
	plt.plot(data['r'], data['mid'], c=color, lw=1.8, label=dwarf)

	plt.xlim(xmin, xmax)
	
	leg = plt.legend(fontsize=18, handlelength=0, handletextpad=0, loc='lower left')
	for item in leg.legendHandles:
		item.set_visible(False)

	plt.savefig('Plots/FM_JA_{}.pdf'.format(dwarf))
	plt.close(fig)

def plot_FM_JR(data, dwarf):
	xlabel, ylabel = r'$r\,\mathrm{[kpc]}$', r'$F_M(r)\,\mathrm{[M}_\odot\,\mathrm{kpc}^{-3}\,\mathrm{km}^{-3}\,\mathrm{s}^{3}\mathrm{]}$'

	color = '#994B92'

	fig = plt.figure(figsize=(5, 5))

	ax = plt.subplot(1, 1, 1)
	plt.sca(ax)

	xmin, xmax = min(data['r']), max(data['r'])

	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	
	plt.fill_between(data['r'], data['2sl'], data['2su'], 
		color=color, alpha=0.2, linewidth=0.0)
	plt.fill_between(data['r'], data['1sl'], data['1su'], 
		color=color, alpha=0.4, linewidth=0.0)
	plt.plot(data['r'], data['1sl'], c=color, lw=0.4, alpha=0.8)
	plt.plot(data['r'], data['1su'], c=color, lw=0.4, alpha=0.8)
	plt.plot(data['r'], data['2sl'], c=color, lw=0.4, alpha=0.3)
	plt.plot(data['r'], data['2su'], c=color, lw=0.4, alpha=0.3)
	plt.plot(data['r'], data['mid'], c=color, lw=1.8, label=dwarf)

	plt.xlim(xmin, xmax)
	
	leg = plt.legend(fontsize=18, handlelength=0, handletextpad=0, loc='lower left')
	for item in leg.legendHandles:
		item.set_visible(False)

	plt.savefig('Plots/FM_JR_{}.pdf'.format(dwarf))
	plt.close(fig)

def plot_dwarfs(load_fn, plot_fn):
	directories = glob('Final_Data/*/')
	for directory in directories:
		dwarf = directory.split('/')[-2]
		data = load_fn(dwarf)
		plot_fn(data, dwarf)

def plot_read_prior(ax):
	purple = '#994B92'
	blue = '#4888B0'
	r, beta = load_read_prior()
	ax.plot(r, beta, c='k', lw=0.8, zorder=0)
	ax.plot(r, np.repeat(0, len(r)), c='k', lw=0.8, zorder=0)
	ax.fill_between(r, np.repeat(0, len(r)), beta, 
		color='k', alpha=0.2, label=r'$\mathrm{Our}\,\mathrm{Prior}$', zorder=0)

def plot_edge_average(ax, files):
	purple = '#994B92'
	blue = '#4888B0'
	arrs = []
	r_min_arr = []
	r_max_arr = []
	fns = []
	for file in files:
		r, _, _, beta = np.loadtxt(file, unpack=True)
		r_min_arr.append(min(r))
		r_max_arr.append(max(r))
		ifn = interpolate_simulation(file)
		fns.append(ifn)
	r_min = max(r_min_arr)
	r_max = min(r_max_arr)
	r_arr = np.geomspace(r_min, 0.9 * r_max, 1000)
	for fn in fns:
		arrs.append(fn(r_arr))
	arrs = np.array(arrs)
	average = np.sum(arrs, axis=0)/len(files)
	variance = np.var(arrs, axis=0)
	ax.plot(r_arr, average, c=purple, lw=1.8, alpha=0.8, label=r'$\mathrm{EDGE}\,\mathrm{Simulation}$')
	ax.fill_between(r_arr, average - np.sqrt(variance), average + np.sqrt(variance), linewidth=0.0, color=purple, alpha=0.15)

def plot_aquarius_average(ax, aqfile):
	purple = '#994B92'
	blue = '#4888B0'
	arrs = []
	r_min_arr = []
	r_max_arr = []
	fns = []
	data = np.loadtxt(aqfile)
	for row in range(1, len(data[:, 0])):
		r, beta = 1e3 * data[0, :], data[row, :]
		r_min_arr.append(min(r))
		r_max_arr.append(max(r))
		ifn = interp1d(r, beta, kind='linear', fill_value='extrapolate')
		fns.append(ifn)
	r_min = max(r_min_arr)
	r_max = min(r_max_arr)
	r_arr = np.geomspace(r_min, 0.9 * r_max, 1000)
	for fn in fns:
		arrs.append(fn(r_arr))
	arrs = np.array(arrs)
	average = np.sum(arrs, axis=0)/(len(data[:, 0]) - 1)
	variance = np.var(arrs, axis=0)
	ax.plot(r_arr, average, c=blue, lw=1.8, ls=(1,(5,1)), alpha=0.8, label=r'$\mathrm{Aquarius}\,\mathrm{Simulation}$', zorder=9)
	ax.fill_between(r_arr, average - np.sqrt(variance), average + np.sqrt(variance), linewidth=0.0, color=blue, alpha=0.15, zorder=9)