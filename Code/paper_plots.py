import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from scipy.interpolate import interp1d, UnivariateSpline
import re
import collections
import csv

from data import get_dwarf, load_rho, load_vesc_JA, load_read_prior, interpolate_simulation, load_FM_JA, load_FG_JA, get_mass_mix, get_mass_mix_nointerp
from plotting import plot_read_prior, plot_edge_average, plot_aquarius_average

def figure_one(rho_load_fn, vesc_load_fn, dwarf, legend_label='Leo II'):

	fig = plt.figure(figsize=(5, 8))
	
	xlabel, ylabel = r'$r\,\mathrm{[kpc]}$', r'$\rho(r)\,\mathrm{[M}_{\odot} \, \mathrm{kpc}^{-3}\mathrm{]}$'

	data = rho_load_fn(dwarf)
	xmin, xmax = min(data['r']), 1e2
	color = '#994B92'
	hlr = 0.233

	ax = plt.subplot(2, 1, 1)
	plt.sca(ax)

	plt.xscale('log')
	plt.yscale('log')
	plt.xlim(xmin, xmax)
	plt.ylim(1e-1,1e11)
	plt.ylabel(ylabel)
	
	plt.fill_between(data['r'], data['2sl'], data['2su'], 
		color=color, alpha=0.2, linewidth=0.0)
	plt.fill_between(data['r'], data['1sl'], data['1su'], 
		color=color, alpha=0.4, linewidth=0.0)
	plt.plot(data['r'], data['1sl'], c=color, lw=0.4, alpha=0.8)
	plt.plot(data['r'], data['1su'], c=color, lw=0.4, alpha=0.8)
	plt.plot(data['r'], data['2sl'], c=color, lw=0.4, alpha=0.3)
	plt.plot(data['r'], data['2su'], c=color, lw=0.4, alpha=0.3)
	plt.plot(data['r'], data['mid'], c=color, lw=1.8)
	
	if dwarf == 'LeoII':
		plt.plot([hlr,hlr], [1e-1, 1e11], c='k', ls='-', lw=0.3)
		plt.text(0.58 * hlr, 3e2 * plt.axis()[2], s=r'$r_{1/2}$' + r'$= {:.2f}\,$'.format(hlr) + r'$\mathrm{kpc}$', fontsize=14, rotation=90)
		ax.set_yticks([1e-1, 1e1, 1e3, 1e5, 1e7, 1e9, 1e11])

	ax.tick_params(axis='both', which='major', pad=3)
	ax.set_xticklabels(['', ''])
	
	xlabel, ylabel = r'$r\,\mathrm{[kpc]}$', r'$v_{\mathrm{esc}}(r)\,\mathrm{[km} \, \mathrm{s}^{-1}\mathrm{]}$'

	data = vesc_load_fn(dwarf)
	color = '#994B92'
	hlr = 0.233

	ax = plt.subplot(2, 1, 2)
	plt.sca(ax)

	plt.xscale('log')
	plt.yscale('linear')
	plt.xlim(xmin, xmax)
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
	plt.plot(data['r'], data['mid'], c=color, lw=1.8, label=legend_label)
	
	leg = plt.legend(fontsize=18, handlelength=0, handletextpad=0, loc='lower left')
	for item in leg.legendHandles:
		item.set_visible(False)
	
	if dwarf == 'LeoII':
		plt.plot([hlr,hlr], [plt.axis()[2], plt.axis()[3]], c='k', ls='-', lw=0.3)

	ax.tick_params(axis='x', which='major', pad=6)
	ax.tick_params(axis='y', which='major', pad=3)
	plt.tight_layout()
	plt.subplots_adjust(hspace=0.07)
	plt.savefig('Plots/rho_vesc_{}.pdf'.format(dwarf))
	plt.close(fig)

def figure_two():
	plt.figure()
	plt.xlabel(r'$r\,\mathrm{[kpc]}$')
	plt.ylabel(r'$\beta_{\mathrm{dm}}(r) := 1 - \sigma_t^2/\sigma_r^2$')
	ax = plt.gca()
	halo_files = glob('Beta_Data/Halo*z0.00_beta.txt')
	plot_read_prior(ax)
	plot_aquarius_average(ax, aqfile='Beta_Data/beta_Aquarius.txt')
	plot_edge_average(ax, halo_files)
	plt.xscale('linear')
	plt.yscale('linear')
	plt.xlim(1e-2, 1e0)
	plt.ylim(-0.5, 1.0)
	plt.legend(fontsize=14, loc='upper left')
	plt.savefig('Plots/beta_plot_linear.pdf')

def figure_three(load_FM_fn, load_FG_fn, dwarf):
	xlabel, ylabel = r'$r\,\mathrm{[kpc]}$', r'$\bar{F}_\mathrm{M}(r), \, \bar{F}^{\mathrm{max}}_{G}(r)\,\mathrm{[M}_{\odot} \, \mathrm{kpc}^{-3}\,\mathrm{km}^{-3}\,\mathrm{s}^3\mathrm{]}$'

	fig = plt.figure(figsize=(5, 5))

	ax = plt.subplot(1, 1, 1)
	plt.sca(ax)

	plt.xscale('log')
	plt.yscale('log')
	plt.xlim(1e-2, 1e2)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

	data = load_FG_fn(dwarf)
	color = '#4888B0'

	plt.fill_between(data['r'], data['2sl'], data['2su'], 
		color=color, alpha=0.2, linewidth=0.0, zorder=9)
	plt.fill_between(data['r'], data['1sl'], data['1su'], 
		color=color, alpha=0.5, linewidth=0.0, zorder=9)
	plt.plot(data['r'], data['1sl'], c=color, lw=0.4, alpha=0.8, zorder=9)
	plt.plot(data['r'], data['1su'], c=color, lw=0.4, alpha=0.8, zorder=9)
	plt.plot(data['r'], data['2sl'], c=color, lw=0.4, alpha=0.3, zorder=9)
	plt.plot(data['r'], data['2su'], c=color, lw=0.4, alpha=0.3, zorder=9)
	plt.plot(data['r'], data['mid'], c=color, lw=1.8, ls=(1, (5, 1)), label=r'$\bar{F}^{\mathrm{max}}_{G}(r)$', zorder=9)

	data = load_FM_fn(dwarf)
	color = '#994B92'
	
	plt.fill_between(data['r'], data['2sl'], data['2su'], 
		color=color, alpha=0.2, linewidth=0.0)
	plt.fill_between(data['r'], data['1sl'], data['1su'], 
		color=color, alpha=0.4, linewidth=0.0)
	plt.plot(data['r'], data['1sl'], c=color, lw=0.4, alpha=0.8)
	plt.plot(data['r'], data['1su'], c=color, lw=0.4, alpha=0.8)
	plt.plot(data['r'], data['2sl'], c=color, lw=0.4, alpha=0.3)
	plt.plot(data['r'], data['2su'], c=color, lw=0.4, alpha=0.3)
	plt.plot(data['r'], data['mid'], c=color, lw=1.8, label=r'$\bar{F}_\mathrm{M}(r)$')
	
	leg = plt.legend(fontsize=14, loc='upper right', markerfirst=False)
	
	hlr = 0.233
	if dwarf == 'LeoII':
		plt.plot([hlr,hlr], [plt.axis()[2], plt.axis()[3]], c='k', ls='-', lw=0.3)
		plt.text(0.57 * hlr, 3e2 * plt.axis()[2], s=r'$r_{1/2}$' + r'$= {:.2f}\,$'.format(hlr) + r'$\mathrm{kpc}$',
				fontsize=14, rotation=90)
		ax.set_yticks([1e-7, 1e-5, 1e-3, 1e-1, 1e1, 1e3, 1e5, 1e7])

	plt.savefig('Plots/FBoth_{}.pdf'.format(dwarf))
	plt.close(fig)

def figure_four(FM, FG):
	color, color2 = '#4888B0', '#994B92'

	plt.figure()

	plt.xlabel(r'$m\,\mathrm{[keV]}$')
	plt.ylabel(r'$f_{\mathrm{max}} \, \mathrm{(in}\,\mathrm{Nat.}\, \mathrm{Units)}$')

	m_arr = np.geomspace(0.1, 10.0, 1000)

	plt.plot(m_arr, 7.86494903545393e-09 * FM / m_arr**4, c=color2, lw=2.4, ls='-', label=r'$\bar{F}_M^{\mathrm{max}}/m^4$')
	plt.plot(m_arr, 7.86494903545393e-09 * FG / m_arr**4, c=color, lw=2.4, ls=(1, (5, 1)), label=r'$\bar{F}_G^{\mathrm{max}}/m^4$')

	plt.plot(m_arr, np.repeat(1 / (2 * np.pi)**3, len(m_arr)), c='k', lw=1.2, ls=(1, (2, 1)))
	plt.text(2.8e0, 0.45e-2, r'Fermi-Dirac', fontsize=12)
	plt.plot(m_arr, 0.093 * 0.1202 * 0.5 * np.power(m_arr, -1.0) / (2 * np.pi)**3, c='k', lw=1.2, ls=(1, (2, 1)))
	plt.text(2e0, 0.9e-5, r'NRP', fontsize=12, rotation=-21)
	plt.legend(fontsize=16, loc='lower left')

	plt.xscale('log')
	plt.yscale('log')

	plt.ylim(1e-6, 1e-1)
	plt.xlim(1e-1, 1e1)

	plt.annotate('', xy=(0.18, 3.2e-4), xytext=(0.4, 1e-3), 
            arrowprops=dict(facecolor='k', alpha=1.0, width=1.0, headwidth=5.0, headlength=5.0),
            )
	plt.text(0.19, 4.3e-4, r'$\mathrm{Excluded}$', rotation=30.5, fontsize=12)

	plt.savefig('Plots/fmax_plot.pdf')

def figure_five():
	blue = '#4888B0'
	purple = '#994B92'

	FD_central = 0.91762955174651
	FD_min1sig = 0.592346941 
	FD_plus1sig = 1.301076756
	FD_min2sig = 0.413609973
	FD_plus2sig = 1.506687845

	bbn_m_arr, bbn_mix_arr = np.loadtxt('Sterile_Data/bbn_bound.txt')[:, 0], np.loadtxt('Sterile_Data/bbn_bound.txt')[:, 1]
	
	mass_mid, mix_mid = get_mass_mix(FD_central, 'Sterile_Data/RP_sterile_Gaussiancoarse_central.txt')
	mass_1sl, mix_1sl = get_mass_mix(FD_min1sig, 'Sterile_Data/RP_sterile_Gaussiancoarse_minus1sigma.txt')
	mass_1su, mix_1su = get_mass_mix(FD_plus1sig, 'Sterile_Data/RP_sterile_Gaussiancoarse_plus1sigma.txt')

	mass_mid_i, mix_mid_i = get_mass_mix_nointerp(FD_central, 'Sterile_Data/RP_sterile_Gaussiancoarse_central.txt')
	mass_1sl_i, mix_1sl_i = get_mass_mix_nointerp(FD_min1sig, 'Sterile_Data/RP_sterile_Gaussiancoarse_minus1sigma.txt')
	mass_1su_i, mix_1su_i = get_mass_mix_nointerp(FD_plus1sig, 'Sterile_Data/RP_sterile_Gaussiancoarse_plus1sigma.txt')

	mass_2sl = [0.413609973, 0.413609973, 5.500000000000000444e-01, 5.500000000000000444e-01]
	mix_2sl = [1e-7, 1.619367485293161006e-10, 1.619367485293161006e-10, 1e-13]

	mass_2su, mix_2su = get_mass_mix(FD_plus2sig, 'Sterile_Data/RP_sterile_Gaussiancoarse_plus2sigma.txt')
	mass_2su_i, mix_2su_i = get_mass_mix_nointerp(FD_plus2sig, 'Sterile_Data/RP_sterile_Gaussiancoarse_plus2sigma.txt')

	over_mass, over_mix = np.loadtxt('Sterile_Data/DM_overproduction.txt', unpack=True)


	xlabel, ylabel = r'$m_{\mathrm{s}}\,\mathrm{[keV]}$', r'$\sin^2 2\theta$'

	plt.figure()
	ymin, ymax = 1e-13, 1e-7
	xmin, xmax = 0.3, max(bbn_m_arr)
	plt.xlim(xmin, xmax)
	plt.ylim(ymin, ymax)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

	plt.plot(bbn_m_arr, bbn_mix_arr, color=purple, lw=1.8, zorder=0)
	plt.fill_between(bbn_m_arr, np.repeat(ymin, len(bbn_m_arr)), bbn_mix_arr, color='w', facecolor='w', edgecolor=purple, hatch='///', linewidths=0.0, alpha=0.4, zorder=0)

	plt.plot(mass_mid, mix_mid, color=blue, lw=1.8, alpha=0.9, zorder=2)
	plt.plot(mass_mid, mix_mid, color='k', lw=1.8, alpha=0.3, zorder=2)

	plt.fill_betweenx(mix_2sl, np.repeat(xmin, len(mass_2sl)), mass_2sl, color='w', facecolor='w', edgecolor=blue, hatch='xxx', linewidths=0.0, alpha=0.3, zorder=1)

	plt.fill(np.append(mass_1sl, mass_1su[::-1]), np.append(mix_1sl, mix_1su[::-1]), alpha=0.7, color="#4888B0", linewidth=0.0)
	plt.fill(np.append(mass_2sl, mass_2su[::-1]), np.append(mix_2sl, mix_2su[::-1]), alpha=0.4, color="#4888B0", linewidth=0.0)
	plt.fill(np.append(mass_2sl, mass_2su[::-1]), np.append(mix_2sl, mix_2su[::-1]), color='none', facecolor='none', edgecolor='w', hatch='xxx', alpha=0.1, zorder=3)

	plt.plot(over_mass, over_mix, lw=1.8, zorder=0, color='k')
	plt.fill_between(over_mass, over_mix, np.repeat(ymax, len(over_mix)), alpha=0.3, color='gray', zorder=0)


	plt.gca().set_zorder(9)
	plt.xscale('log')
	plt.yscale('log')
	plt.text(3.05, 4.5e-13, 'BBN', rotation=-21, color='purple', fontsize=12)
	plt.annotate('', xy=(0.5, 1.09e-11), xytext=(0.9, 1e-10), 
            arrowprops=dict(facecolor='k', alpha=1.0, width=1.0, headwidth=5.0, headlength=5.0),
            )
	plt.text(5, 5e-10, r'$\Omega_{\mathrm{s}} > \Omega_{\mathrm{dm}}$', rotation=-24, color='k', fontsize=12)
	plt.text(0.52, 2.1e-11, r'$\mathrm{Excluded}$', rotation=45, fontsize=12)
	plt.text(1.7, 1.7e-11, 'Phase-Space', color=blue, rotation=-52, fontsize=12)
	plt.savefig('Plots/RP_sterile.pdf')

def figure_six():
	linestyles = [(0, (1,1)), (0, (3, 1, 1, 1)), (0, (1,3)), (0, (3,4)), (0, (3,1.2)), (0, (3, 1, 1, 1, 1, 1))]

	with open("Summary_Data/mdeg_JA_inner.csv") as fp:
	    reader = csv.reader(fp, delimiter=",", quotechar='"')
	    next(reader, None) 
	    data_deg = [row for row in reader]

	with open("Summary_Data/mFD_FG_inner.csv") as fp:
	    reader = csv.reader(fp, delimiter=",", quotechar='"')
	    next(reader, None) 
	    data_FD_Gauss = [row for row in reader]

	with open("Summary_Data/mFD_FM_JA_inner.csv") as fp:
	    reader = csv.reader(fp, delimiter=",", quotechar='"')
	    next(reader, None) 
	    data_FD_Max = [row for row in reader]

	with open("Summary_Data/mNRP_FG_inner.csv") as fp:
	    reader = csv.reader(fp, delimiter=",", quotechar='"')
	    next(reader, None) 
	    data_NRP_Gauss = [row for row in reader]

	with open("Summary_Data/mNRP_FM_JA_inner.csv") as fp:
	    reader = csv.reader(fp, delimiter=",", quotechar='"')
	    next(reader, None) 
	    data_NRP_Max = [row for row in reader]

	dwarf_names = [r"$\mathrm{UMi}$",r"$\mathrm{Sextans}$",r"$\mathrm{Seg\, I}$",r"$\mathrm{Sculptor}$", r"$\mathrm{Leo\, II}$", r"$\mathrm{Leo\, I}$", r"$\mathrm{Fornax}$", r"$\mathrm{Draco}$",r"$\mathrm{Carina}$", r"$\mathrm{CVnl}$", r"$\mathrm{And21}$"]

	data_deg = np.array(data_deg)
	data_deg = np.array(data_deg)[np.argsort(data_deg[:,0]),1:].astype(np.float)[::-1]
	data_FD_Gauss = np.array(data_FD_Gauss)
	data_FD_Gauss = np.array(data_FD_Gauss)[np.argsort(data_FD_Gauss[:,0]),1:].astype(np.float)[::-1]
	data_FD_Max = np.array(data_FD_Max)
	data_FD_Max = np.array(data_FD_Max)[np.argsort(data_FD_Max[:,0]),1:].astype(np.float)[::-1]
	data_NRP_Gauss = np.array(data_NRP_Gauss)
	data_NRP_Gauss = np.array(data_NRP_Gauss)[np.argsort(data_NRP_Gauss[:,0]),1:].astype(np.float)[::-1]
	data_NRP_Max = np.array(data_NRP_Max)
	data_NRP_Max = np.array(data_NRP_Max)[np.argsort(data_NRP_Max[:,0]),1:].astype(np.float)[::-1]

	############################## controls ##############################

	ytickpos = [1,2,3,4,5,6,7,8,9,10,11]

	# to remove
	central = 0.4
	lower1sigma = 0.3
	upper1sigma = 0.5
	lower2sigma = 0.2
	upper2sigma = 0.6
	# 

	alphacentral = 1
	alpha1sigma = 0.7
	alpha2sigma = 0.3
	alphaseparator = 0.2

	width = 0.07

	textsize = 10
	textoffset = 0.11


	############################## plots ##############################

	plt.figure(figsize=(12,8))

	ax1 = plt.subplot(1,3,1)
	ax2 = plt.subplot(1,3,2)
	ax3 = plt.subplot(1,3,3)

	ax1.tick_params(axis='both', which='major', labelsize=18)
	ax1.tick_params(axis='both', which='minor', labelsize=18)
	ax2.tick_params(axis='both', which='major', labelsize=18)
	ax2.tick_params(axis='both', which='minor', labelsize=18)
	ax3.tick_params(axis='both', which='major', labelsize=18)
	ax3.tick_params(axis='both', which='minor', labelsize=18)


	ax1.set_yticks(ytickpos)
	ax1.set_yticklabels(dwarf_names)
	ax1.tick_params(axis='y', which='both', left=None,right=None)
	ax1.set_ylim(ymin=0.5,ymax=11.5)
	ax1.set_xlim(xmin=0.02,xmax=2)
	ax1.set_title(r"$\mathrm{Pauli\ Exclusion\ Principle}$", fontsize=16)
	ax1.set_xlabel(r"$m_\mathrm{deg}\ [\mathrm{keV}]$")
	ax1.semilogx()
	ax1.set_xticks([0.1, 1])
	ax1.set_xticklabels([r'$0.1$',r'$1.0$'])
	ax1.axvline(0.1,color="black",alpha=0.1,linewidth=0.5)
	ax1.axvline(1,color="black",alpha=0.1,linewidth=0.5)

	ax2.set_yticks(ytickpos)
	ax2.set_yticklabels([])
	ax2.tick_params(axis='y', which='both', left=None,right=None)
	ax2.set_ylim(ymin=0.5,ymax=11.5)
	ax2.set_xlim(xmin=0.015,xmax=4)
	ax2.set_title(r"$\mathrm{FD\ Statistics}$", fontsize=16)
	ax2.set_xlabel(r"$m_\mathrm{FD}\ [\mathrm{keV}]$")
	ax2.semilogx()
	ax2.set_xticks([0.1, 1])
	ax2.set_xticklabels([r'$0.1$',r'$1.0$'])
	ax2.axvline(0.1,color="black",alpha=0.1,linewidth=0.5)
	ax2.axvline(1,color="black",alpha=0.1,linewidth=0.5)

	ax3.set_yticks(ytickpos)
	ax3.set_yticklabels([])
	ax3.tick_params(axis='y', which='both', left=None,right=None)
	ax3.set_ylim(ymin=0.5,ymax=11.5)
	ax3.set_xlim(xmin=0.05,xmax=40)
	ax3.set_title(r"$\mathrm{NRP\ Sterile\ Neutrinos}$", fontsize=16)
	ax3.set_xlabel(r"$m_\mathrm{NRP}\ [\mathrm{keV}]$")
	ax3.semilogx()
	ax3.set_xticks([0.1, 1, 10])
	ax3.set_xticklabels([r'$0.1$',r'$1.0$',r'$10$'])
	ax3.axvline(0.1,color="black",alpha=0.1,linewidth=0.5)
	ax3.axvline(1,color="black",alpha=0.1,linewidth=0.5)
	ax3.axvline(10,color="black",alpha=0.1,linewidth=0.5)

	plt.subplots_adjust(wspace = 0.15)

	grad_colors = ['#4888B0', '#5082AD', '#587CAA', '#6076A7', '#6870A4', '#716AA1', '#79639E', '#815D9B', '#895798', '#915195', '#994B92']

	for ax in [ax1,ax2,ax3]:
	    for num in ytickpos:
	        color1 = grad_colors[11 - int(num)]
	        color2 = grad_colors[10]
	        color3 = grad_colors[0]

	        if ax in [ax2, ax3]:

	            offset = 0.15

	            if ax == ax2:
	                data_Max = data_FD_Max
	                data_Gauss = data_FD_Gauss
	            else:
	                data_Max = data_NRP_Max
	                data_Gauss = data_NRP_Gauss

	            # max-coarse
	            ax.fill_between([data_Max[num-1,1], data_Max[num-1,4]], num+offset+width, num+offset-width, color=color2,alpha=alpha2sigma, linewidths=0.0)
	            ax.fill_between([data_Max[num-1,2], data_Max[num-1,3]], num+offset+width, num+offset-width, color=color2,alpha=alpha1sigma, linewidths=0.0)
	            ax.fill_between([data_Max[num-1,0], data_Max[num-1,0]], num+offset+0.7*width, num+offset-0.7*width, color="black",alpha=0.6)
	            ax.fill_between([data_Max[num-1,0], data_Max[num-1,0]], num+offset+0.7*width, num+offset-0.7*width, color=color2,alpha=0.9)

	            
	            ax.axhline(num, color="black", alpha=alphaseparator, linestyle=linestyles[0])

	            # gaussian coarse
	            ax.fill_between([data_Gauss[num-1,1], data_Gauss[num-1,4]], num-offset+width, num-offset-width, color=color3,alpha=alpha2sigma, linewidths=0.0)
	            ax.fill_between([data_Gauss[num-1,2], data_Gauss[num-1,3]], num-offset+width, num-offset-width, color=color3,alpha=alpha1sigma, linewidths=0.0)
	            ax.fill_between([data_Gauss[num-1,0], data_Gauss[num-1,0]], num-offset+0.7*width, num-offset-0.7*width, color="black",alpha=0.6)
	            ax.fill_between([data_Gauss[num-1,0], data_Gauss[num-1,0]], num-offset+0.7*width, num-offset-0.7*width, color=color3,alpha=0.9)

	        else:
	            ax.fill_between([data_deg[num-1,1], data_deg[num-1,4]], num+width, num-width, color=color1,alpha=alpha2sigma, linewidths=0.0)
	            ax.fill_between([data_deg[num-1,2], data_deg[num-1,3]], num+width, num-width, color=color1,alpha=alpha1sigma, linewidths=0.0)
	            ax.fill_between([data_deg[num-1,0], data_deg[num-1,0]], num+0.7*width, num-0.7*width, color="black",alpha=0.6)
	            ax.fill_between([data_deg[num-1,0], data_deg[num-1,0]], num+0.7*width, num-0.7*width, color=color1,alpha=0.9)


	        if ax == ax2:
	            ax.text(0.0162, num + textoffset, r'$\mathrm{Max.}$', fontsize=textsize, alpha=0.7, weight="bold")
	            ax.text(0.0162, num - 2*textoffset, r'$\mathrm{Gauss.}$', fontsize=textsize, alpha=0.7, weight="bold")

	plt.savefig("Plots/Bounds_summary.pdf")


if __name__ == '__main__':
	directories = glob('Final_Data/*/')

	# Make Fig. 1
	'''
	for directory in directories:
		dwarf = get_dwarf(directory)
		if dwarf != 'LeoII':
			figure_one(load_rho, load_vesc_JA, dwarf, legend_label=dwarf)
		else:
			figure_one(load_rho, load_vesc_JA, dwarf)
	'''

	# Make Fig. 2
	'''
	figure_two()
	'''

	# Make Fig. 3
	'''
	for directory in directories:
		dwarf = get_dwarf(directory)
		figure_three(load_FM_JA, load_FG_JA, dwarf)
	'''

	# Make Fig. 4
	'''
	figure_four(FM=5639.810392284446, FG=363441.1077239243)
	'''

	# Make Fig. 5
	
	figure_five()
	

	# Make Fig. 6
	'''
	figure_six()
	'''

