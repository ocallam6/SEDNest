
import numpy as np
import minimint
def _systematic_between_atlas_phoenix_linear(sed_atlas,sed_phoenix):
    teff=np.linspace(4000,10000,5)
    mh=np.linspace(-2,0.5,5)
    logg=np.linspace(2.0,5,5)
    grid=np.meshgrid(teff,mh,logg)
    residuals=[]
    teff,mh,logg=grid[0].flatten(),grid[1].flatten(),grid[2].flatten()
    for i in range(len(teff)):
        
            change=(sed_atlas.photometry_vega(mh[i],teff[i],logg[i],radius=1,distance=10,a0=1,Rv=3.1)-sed_phoenix.photometry_vega(mh[i],teff[i],logg[i],radius=1,distance=10,a0=1,Rv=3.1))
            if(np.product([np.abs(change[0][i])<np.inf for i in range(8)])):

                residuals.append((change[0]))
    return residuals
import pandas as pd    
def target_systematic_between_atlas_phoenix_linear(sed_atlas,sed_phoenix):
    age=np.linspace(8.8,10.3,10)
    feh=np.linspace(-1,0.5,20)
    mass=np.linspace(0.4,1.1,100)
    grid=np.meshgrid(age,feh,mass)
    age,feh,mass=grid[0].flatten(),grid[1].flatten(),grid[2].flatten()

    ii=minimint.Interpolator(filts=['Gaia_G_EDR3','Gaia_BP_EDR3'])
    
    ii=pd.DataFrame(ii(mass=mass,feh=feh,logage=age))
    teff=10**ii['logteff'].values
    mh=ii['feh'].values
    logg=ii['logg'].values
    g=ii[['Gaia_G_EDR3','Gaia_BP_EDR3']].values
    residuals=[]
    gs=[]
    for i in range(len(teff)):
        
            change=(sed_atlas.photometry_vega(mh[i],teff[i],logg[i],radius=1,distance=10,a0=0,Rv=3.1)[0][0:2]-g[i])#-sed_phoenix.photometry_vega(mh[i],teff[i],logg[i],radius=1,distance=10,a0=0,Rv=3.1))
            change=change[0]-change[1]
            #if(np.product([np.abs(change[0][i])<np.inf for i in range(2)])):
            if(np.abs(change)<np.inf):
                residuals.append((change))
                
    return residuals,gs
    