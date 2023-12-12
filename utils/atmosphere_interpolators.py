from glob import glob
from dustapprox.io import svo
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import RegularGridInterpolator
import astropy.units as u
import glob
# Model for Creating Interpolators for Spectra.

def atlas_linear():

    loc='/Users/mattocallaghan/VaeStar/Isochrones_data/Kurucz2003all/*.fl.dat.txt'
    models = glob.glob(loc)
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
    wavelength=wl[0] 
                       
    return [interp,u.AA,flux_unit,np.array(wavelength)]

def atlas_cubic():

    loc='/Users/mattocallaghan/VaeStar/Isochrones_data/Kurucz2003all/*.fl.dat.txt'
    models = glob.glob(loc)
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
    
    interp=RegularGridInterpolator((feh_unique,logg_unique,teff_unique),np.nan_to_num(flux_grid,nan=0.0),method='cubic',
                                    bounds_error=False, fill_value=0.0)  
    wavelength=wl[0] 
                       
    return [interp,u.AA,flux_unit,np.array(wavelength)]

#####
#Phoenix
#####

folder_path = '/Users/mattocallaghan/SEDNest/Data/Phoenix/'
file_pattern = '*.fits'
import pandas as pd
import fitsio as fits
import astropy.io.fits
import pickle
import glob
wavelength_cut=706247

def _phoenix_to_csv():
 # Get a list of all FITS files in the folder
    fits_files = glob.glob(folder_path + file_pattern)
    def fitsread(fits_file,columns):
        try: 
            par=fits.read(fits_file,columns=columns)
            return par, columns
        except:
            columns=columns[:-1]
            return fitsread(fits_file,columns)

    wavelength_cut=706247
    #This is the one with the smaller wavelenghts. We need this because solar isochrones have many sampled points and we want the same
    par1=fits.read('/Users/mattocallaghan/SEDNest/Data/Phoenix/phoenixm05_5000.fits',columns=['WAVELENGTH'])
    temp_file1=astropy.io.fits.open('/Users/mattocallaghan/SEDNest/Data/Phoenix/phoenixm05_5000.fits')
    wavelengths=pd.DataFrame(np.stack(np.array(par1,dtype=object)).astype(float),columns=['WAVELENGTH'])
    wavelengths=wavelengths[wavelengths['WAVELENGTH'].values<wavelength_cut].values[:,0] # weird shape

    tests=[]
    lens=[]
    dfs=[]
    teffs=[]
    fehs=[]
    wl=[]
    fluxes=[]
    logg_names = {'g00':0.0,'g05':0.5,'g10':1.0,'g15':1.5,'g20':2.0,'g25':2.5,'g30':3.0,'g35':3.5,'g40':4.0,'g45':4.5,'g50':5.0,'g55':5.5,'60':6.0}
    # Open each FITS file
    flag=True
    for fits_file in fits_files:
        temp_file1=astropy.io.fits.open(fits_file)
        if(float(temp_file1[0].header['TEFF'])<10000):
            columns=['WAVELENGTH','g00','g05','g10','g15','g20','g25','g30','g35','g40','g45','g50','g55','60']

            spectrum,columns=fitsread(fits_file,columns=columns)
            
            
            spectrum=np.stack(np.array(spectrum,dtype=object)).astype(float)

            spectrum=pd.DataFrame(np.stack(np.array(spectrum,dtype=object)),columns=[columns][::-1]).astype(float)
            spectrum=spectrum[spectrum['WAVELENGTH'].values<wavelength_cut] # this cut should be fine for us becasue of our passbands

            
            if(len(spectrum)!=len(wavelengths)):
                new_spectrum=pd.DataFrame(wavelengths,columns=[columns[0]])
                for column in columns[1:]:
                    cs = CubicSpline(spectrum['WAVELENGTH'].values.reshape(len(spectrum)), spectrum[column].values[:,0])
                    new_spectrum[column]=cs(wavelengths) # weird shape of column values
                    
                spectrum=new_spectrum



            for column in columns[1:]:
                dict_pars={'LOGG':logg_names[column],'TEFF':float(temp_file1[0].header['TEFF']),
                                    'WL_UNIT':temp_file1[1].header['TUNIT1'],'FLUX_UNIT':temp_file1[1].header['TUNIT2']}
                new_spectrum=pd.DataFrame(dict_pars,index=[0])

                #new_spectrum['FLUX']=spectrum[column].values #weird shape
                #new_spectrum['LOGG']=logg_names[column]
                #new_spectrum['TEFF']=float(temp_file1[0].header['TEFF'])
                #new_spectrum['WL_UNIT']=temp_file1[1].header['TUNIT1']
                #new_spectrum['FLUX_UNIT']=temp_file1[1].header['TUNIT2']
                
                if(flag):
                    fluxes=spectrum[column].values.astype(float).reshape(len(spectrum))[None,:]
                    flag=False
                else:
                    f=spectrum[column].astype(float).values.reshape(len(spectrum))[None,:]
                    fluxes=np.concatenate([fluxes,f],axis=0)
                

                #fluxes.append(spectrum[column].values)

                teffs.append(temp_file1[0].header['TEFF'])
                
                try:
                    new_spectrum['MH']=float(temp_file1[0].header['METAL10'])
                    fehs.append(temp_file1[0].header['METAL10'])
                except:
                    new_spectrum['MH']=float(temp_file1[0].header['LOG_Z'])
                    fehs.append(temp_file1[0].header['LOG_Z'])
                dfs.append(new_spectrum.reset_index(drop=True))

    spectra=pd.concat(dfs)
    sp_zero_flux=spectra#[spectra['FLUX']>0]
    sp_zero_flux.to_csv('../Data/phoenix_with_cuts')

    with open('../Data/phoenix_fluxes', 'wb') as file:
            pickle.dump(fluxes, file)

def phoenix_linear():
    _phoenix_to_csv()
    sp_zero_flux=pd.read_csv('../Data/phoenix_with_cuts')
    par1=fits.read('/Users/mattocallaghan/SEDNest/Data/Phoenix/phoenixm05_5000.fits',columns=['WAVELENGTH'])
    wavelengths=pd.DataFrame(np.stack(np.array(par1,dtype=object)).astype(float),columns=['WAVELENGTH'])
    wavelengths=wavelengths[wavelengths['WAVELENGTH'].values<wavelength_cut].values[:,0] # weird shape
    pars=np.stack([sp_zero_flux['MH'].astype(float).values,sp_zero_flux['LOGG'].astype(float).values,sp_zero_flux['TEFF'].astype(float).values]).T
    feh_unique=np.unique(sp_zero_flux['MH'].astype(float).values)
    logg_unique=np.unique(sp_zero_flux['LOGG'].astype(float).values)
    teff_unique=np.unique(sp_zero_flux['TEFF'].astype(float).values)
    with open('../Data/phoenix_fluxes', 'rb') as file:
        fluxes = pickle.load(file)

    pars=np.stack([sp_zero_flux['MH'].astype(float).values,sp_zero_flux['LOGG'].astype(float).values,sp_zero_flux['TEFF'].astype(float).values]).T
    feh_unique=np.unique(sp_zero_flux['MH'].astype(float).values)
    logg_unique=np.unique(sp_zero_flux['LOGG'].astype(float).values)
    teff_unique=np.unique(sp_zero_flux['TEFF'].astype(float).values)
    #fluxes=sp_zero_flux['FLUX'].astype(float).values
    parameters=np.stack(np.meshgrid(feh_unique,logg_unique,teff_unique,indexing='ij'),axis=-1)
    flux_grid=(np.zeros((parameters.shape[:-1]+(len(wavelengths),)))*np.NaN)
    indices=[]
    for i,j,k in np.ndindex((parameters.shape[0:-1])):

        idx=np.where([(np.prod(pars[_]==parameters[i,j,k])) for _ in range(len(pars))])[0]
        if(len(idx)>0):
            f=fluxes[int(idx)]
            try:
                flux_grid[i,j,k]=fluxes[int(idx)][:,0]
                indices.append(np.array([i,j,k]))
            except:
                flux_grid[i,j,k]=fluxes[int(idx)][:]
                indices.append(np.array([i,j,k]))

        
    
    interp=RegularGridInterpolator((feh_unique,logg_unique,teff_unique),flux_grid,method='linear',
                                    bounds_error=False, fill_value=np.NaN)  
    wavelength=wavelengths# convert to nm
    return [interp,u.AA,sp_zero_flux['FLUX_UNIT'].values[0],wavelength]



def phoenix_cubic():
    _phoenix_to_csv()
    sp_zero_flux=pd.read_csv('../Data/phoenix_with_cuts')
    par1=fits.read('/Users/mattocallaghan/SEDNest/Data/Phoenix/phoenixm05_5000.fits',columns=['WAVELENGTH'])
    wavelengths=pd.DataFrame(np.stack(np.array(par1,dtype=object)).astype(float),columns=['WAVELENGTH'])
    wavelengths=wavelengths[wavelengths['WAVELENGTH'].values<wavelength_cut].values[:,0] # weird shape
    pars=np.stack([sp_zero_flux['MH'].astype(float).values,sp_zero_flux['LOGG'].astype(float).values,sp_zero_flux['TEFF'].astype(float).values]).T
    feh_unique=np.unique(sp_zero_flux['MH'].astype(float).values)
    logg_unique=np.unique(sp_zero_flux['LOGG'].astype(float).values)
    teff_unique=np.unique(sp_zero_flux['TEFF'].astype(float).values)
    with open('../Data/phoenix_fluxes', 'rb') as file:
        fluxes = pickle.load(file)

    pars=np.stack([sp_zero_flux['MH'].astype(float).values,sp_zero_flux['LOGG'].astype(float).values,sp_zero_flux['TEFF'].astype(float).values]).T
    feh_unique=np.unique(sp_zero_flux['MH'].astype(float).values)
    logg_unique=np.unique(sp_zero_flux['LOGG'].astype(float).values)
    teff_unique=np.unique(sp_zero_flux['TEFF'].astype(float).values)
    #fluxes=sp_zero_flux['FLUX'].astype(float).values
    parameters=np.stack(np.meshgrid(feh_unique,logg_unique,teff_unique,indexing='ij'),axis=-1)
    flux_grid=(np.zeros((parameters.shape[:-1]+(len(wavelengths),)))*np.NaN)
    indices=[]
    for i,j,k in np.ndindex((parameters.shape[0:-1])):

        idx=np.where([(np.prod(pars[_]==parameters[i,j,k])) for _ in range(len(pars))])[0]
        if(len(idx)>0):
            f=fluxes[int(idx)]
            try:
                flux_grid[i,j,k]=fluxes[int(idx)][:,0]
                indices.append(np.array([i,j,k]))
            except:
                flux_grid[i,j,k]=fluxes[int(idx)][:]
                indices.append(np.array([i,j,k]))

        
    
    interp=RegularGridInterpolator((feh_unique,logg_unique,teff_unique),np.nan_to_num(flux_grid,nan=0.0),method='cubic',
                                    bounds_error=False, fill_value=0.0)  
    wavelength=wavelengths# convert to nm
    return [interp,u.AA,sp_zero_flux['FLUX_UNIT'].values[0],wavelength]