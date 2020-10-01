import pandas as pd
import numpy as np

from data import load_summary_data

def mdeg(rho, vesc, g=2):
	return np.power(6 * np.pi**2 * rho / (g * vesc**3) * 7.86494903545393e-09, 0.25)

def mFD(F, g=2):
	return np.power(F * 7.86494903545393e-09 * 2 * (2 * np.pi)**3 / g, 0.25)

def mNRP(F, g=2):
	return np.power(F * 7.86494903545393e-09 * 2 * (2 * np.pi)**3 / (0.1202 * 0.093), (1/3))

def get_mdeg_df(rho_df, vesc_df):
	mdeg_df = pd.DataFrame({})
	mdeg_df['dwarf'] = rho_df.index.values
	mdeg_df['mdeg/keV'] = mdeg(rho_df['rho/Msolkpc^-3'].values, vesc_df['vesc/kms^-1'].values)
	mdeg_df['2sl'] = mdeg(rho_df['2sl'].values, vesc_df['2su'].values)
	mdeg_df['1sl'] = mdeg(rho_df['1sl'].values, vesc_df['1su'].values)
	mdeg_df['1su'] = mdeg(rho_df['1su'].values, vesc_df['1sl'].values)
	mdeg_df['2su'] = mdeg(rho_df['2su'].values, vesc_df['2sl'].values)
	mdeg_df = mdeg_df.set_index('dwarf')
	return mdeg_df

def get_mFD_df(F_df):
	mFD_df = pd.DataFrame({})
	mFD_df['dwarf'] = F_df.index.values
	if 'FM/Msols^3km^-3kpc^-3' in F_df.columns:
		mFD_df['mFD/keV'] = mFD(F_df['FM/Msols^3km^-3kpc^-3'].values)
	else:
		mFD_df['mFD/keV'] = mFD(F_df['FG/Msols^3km^-3kpc^-3'].values)
	mFD_df['2sl'] = mFD(F_df['2sl'].values)
	mFD_df['1sl'] = mFD(F_df['1sl'].values)
	mFD_df['1su'] = mFD(F_df['1su'].values)
	mFD_df['2su'] = mFD(F_df['2su'].values)
	mFD_df = mFD_df.set_index('dwarf')
	return mFD_df

def get_mNRP_df(F_df):
	mNRP_df = pd.DataFrame({})
	mNRP_df['dwarf'] = F_df.index.values
	if 'FM/Msols^3km^-3kpc^-3' in F_df.columns:
		mNRP_df['mFD/keV'] = mNRP(F_df['FM/Msols^3km^-3kpc^-3'].values)
	else:
		mNRP_df['mFD/keV'] = mNRP(F_df['FG/Msols^3km^-3kpc^-3'].values)
	mNRP_df['2sl'] = mNRP(F_df['2sl'].values)
	mNRP_df['1sl'] = mNRP(F_df['1sl'].values)
	mNRP_df['1su'] = mNRP(F_df['1su'].values)
	mNRP_df['2su'] = mNRP(F_df['2su'].values)
	mNRP_df = mNRP_df.set_index('dwarf')
	return mNRP_df

if __name__ == '__main__':
	# Load summary data
	rho_df = load_summary_data('rho_inner.csv')
	vesc_JA_df = load_summary_data('vesc_JA_inner.csv')
	vesc_JR_df = load_summary_data('vesc_JR_inner.csv')
	FM_JA_df = load_summary_data('FM_JA_inner.csv')
	FM_JR_df = load_summary_data('FM_JR_inner.csv')
	FG_df = load_summary_data('FG_inner.csv')

	# Compute degenerate bounds
	'''
	mdeg_JA = get_mdeg_df(rho_df, vesc_JA_df)
	mdeg_JA.to_csv('Summary_Data/mdeg_JA_inner.csv', index=True)
	print(mdeg_JA)
	mdeg_JR = get_mdeg_df(rho_df, vesc_JR_df)
	mdeg_JR.to_csv('Summary_Data/mdeg_JR_inner.csv', index=True)
	print(mdeg_JR)
	'''

	# Compute FD bounds
	'''
	mFD_FM_JA = get_mFD_df(FM_JA_df)
	mFD_FM_JA.to_csv('Summary_Data/mFD_FM_JA_inner.csv', index=True)
	print(mFD_FM_JA)
	mFD_FM_JR = get_mFD_df(FM_JR_df)
	mFD_FM_JR.to_csv('Summary_Data/mFD_FM_JR_inner.csv', index=True)
	print(mFD_FM_JR)
	mFD_FG = get_mFD_df(FG_df)
	mFD_FG.to_csv('Summary_Data/mFD_FG_inner.csv', index=True)
	print(mFD_FG)
	'''

	# Compute NRP bounds
	'''
	mNRP_FM_JA = get_mNRP_df(FM_JA_df)
	mNRP_FM_JA.to_csv('Summary_Data/mNRP_FM_JA_inner.csv', index=True)
	print(mNRP_FM_JA)
	mNRP_FM_JR = get_mNRP_df(FM_JR_df)
	mNRP_FM_JR.to_csv('Summary_Data/mNRP_FM_JR_inner.csv', index=True)
	print(mNRP_FM_JR)
	mNRP_FG = get_mNRP_df(FG_df)
	mNRP_FG.to_csv('Summary_Data/mNRP_FG_inner.csv', index=True)
	print(mNRP_FG)
	'''
