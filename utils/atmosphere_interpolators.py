from glob import glob
from dustapprox.io import svo
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import RegularGridInterpolator
import astropy.units as u
# Model for Creating Interpolators for Spectra.

def atlas_linear():

    loc='/Users/mattocallaghan/VaeStar/Isochrones_data/Kurucz2003all/*.fl.dat.txt'
    models = glob(loc)
    apfields = ['teff', 'logg', 'feh', 'alpha']
    wl=[]
    fehs=[]
    logg=[]
    teff=[]
    fluxes=[]

    label = 'teff={teff:4g} K, logg={logg:0.1g} dex, [Fe/H]={feh:0.1g} dex'
    for fname in models:
        data = svo.spectra_file_reader(fname)
        lamb_unit, flux_unit = svo.get_svo_sprectum_units(data)
        lamb = data['data']['WAVELENGTH'].values * lamb_unit
        truth=(data['data']['WAVELENGTH']<60000)*(data['data']['WAVELENGTH']>2000)
        data['data'][truth]
        if(data['alpha']['value']==0.0):
            wl.append(data['data'][truth]['WAVELENGTH'].values)#.reshape(-1, 2).mean(-1))
            fluxes.append(data['data'][truth]['FLUX'].values)#.reshape(-1, 2).mean(-1))
            fehs.append(data['feh']['value'])
            logg.append(data['logg']['value'])
            teff.append(data['teff']['value'])
            #flux=flux*curves[1](data['data']['WAVELENGTH'].values * lamb_unit,av,Rv=Rv)
    
    pars=np.stack([np.array(fehs),np.array(logg),np.array(teff)]).T
    feh_unique=np.unique(np.array(fehs))
    logg_unique=np.unique(np.array(logg))
    teff_unique=np.unique(np.array(teff))
    parameters=np.stack(np.meshgrid(feh_unique,logg_unique,teff_unique,indexing='ij'),axis=-1)
    flux_grid=(np.zeros((parameters.shape[:-1]+(len(wl[0]),)))*np.NaN)
    indices=[]
    for i,j,k in np.ndindex((parameters.shape[0:-1])):
        try:
            idx=np.where([(np.prod(pars[_]==parameters[i,j,k])) for _ in range(len(pars))])[0]
            
            flux_grid[i,j,k]=fluxes[int(idx)]
            indices.append(np.array([i,j,k]))
        except:
            continue
    interp=RegularGridInterpolator((feh_unique,logg_unique,teff_unique),flux_grid,method='linear',
                                    bounds_error=False, fill_value=np.NaN)  
    wavelength=lamb_unit.to(u.nm)*wl[0] 
                       
    return [interp,u.nm,flux_unit,np.array(wavelength)]
atlas_linear()

