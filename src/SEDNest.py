#Class to implement the SED Fit
import warnings
import minimint # for isochrone theory interpolator
import astropy
import extinction as ext
from dustapprox.io import svo
import pickle
import sys
import numpy as np
from astropy.units import R_sun,pc
sys.path.append('../')
from utils import atmosphere_interpolators
class SEDNest:
    def __init__(self,filters='default',use_isochrones=False,*args, **kwargs):
        '''
        Create a Class of the SED Nested sampling fitter. 
        The extinction law must be taken from the extinction python package
        We use the dustapprox/pyphot to take in passbands and the spectra from SVO.

        '''

        # Initiall the parameters are none. These will be changed depending on whether you are using Isochrones or not.
        self.params=None
        self.use_isochrones=use_isochrones
        # Isochrone and Dustapprox Filters.
        # Passband names will be homogenised off SVO
        self.dustapprox_filters,self.isochrone_filters = self._initialise_filter_names(filters,self.use_isochrones)      
        self.loc_passbands='../Data/passbands' #where pickle passbands are stored
        self.passband_names_location='../Data/passband_names.txt' #names to keep track of what is stored in the pickle
        self.passbands=self._initialise_transmission_functions() #the passbands themselves. 
  
        #Atmosphere Model Interpolatrion Scheme
        #todo, bring in so that will store and save unique interpolation models etc.
        self.atmosphere_names_location='../Data/atmosphere_names.txt'
        self.interpolator_location='../Data/model_atmosphere_interpolator'
        self.lamb_unit_location='../Data/lamb_unit.txt'
        self.flux_unit_location='../Data/flux_unit.txt'
        self.wavelength_location='../Data/wavelength.npy'
        self.interpolation_details=['ATLAS9','LINEAR']
        self.atmosphere_interpolator,self.atmosphere_lamb_unit,self.atmosphere_flux_unit,self.wavelength=self._initialise_interpolator(**kwargs)
    
        #Flux preparation
        if(self.passbands[0].wavelength_unit==str(self.atmosphere_lamb_unit)):
            self.transmission_arrays=np.array([np.nan_to_num(self.passbands[passband](self.wavelength*self.atmosphere_lamb_unit,1*self.atmosphere_flux_unit),nan=0.0) for passband in range(len(self.dustapprox_filters))])
            self.denominator_flux=np.array([np.trapz(x=self.wavelength,y=self.wavelength*self.transmission_arrays[i]) for i in range(len(self.transmission_arrays))])
            self.vega_zero_fluxes=np.array([passband.Vega_zero_flux.value for passband in self.passbands])
        else:
            raise NotImplementedError
        # Extinction Law
        self.extinction_law=self._initialise_extinction_law()
    
    def vega_flux(self,atmosphere_fluxes,radius,distance,a0,Rv):
        '''
        Calculate the Vega flux using the Dustapprox/Pyphot method
        For arbitrary input of fluxes
        Uses an extinction law from the extinciton package
        Fluxes need to be in correct unit
        '''
        if(str(self.passbands[0].wavelength_unit)==str(self.atmosphere_lamb_unit)):
            if(np.array(distance).shape==()):
                radius,distance,a0=np.array(radius).reshape(1),np.array(distance).reshape(1),np.array(a0).reshape(1)

            flux_values=(atmosphere_fluxes*(radius[:,None]*R_sun)**2/(distance[:,None]*pc)**2)
            extincted_flux=np.array(flux_values*(10**(-0.4*np.matmul(a0[:,None],self.extinction_law(self.wavelength,1.0,Rv,'aa')[None,:]))))


            numerator=(self.wavelength[None,:]*extincted_flux)[:,:,None]*self.transmission_arrays.T[None,:,:]
            return (np.trapz(y=numerator,x=self.wavelength[None,:]*self.atmosphere_lamb_unit,axis=1)/self.denomenator)/self.vega_zero_fluxes
        else:
            print('Unit error!!!')
            raise NotImplementedError

    def interpolate_spectra(self,feh,teff,logg):
        '''
        Allows multiple values of the input parameters
        Output will be (n_sets_param,m_fluxes)
        '''
        return np.nan_to_num(self.atmosphere_interpolator((feh,logg,teff)),nan=0.0)

    def _initialise_extinction_law(self,**kwargs):
        return kwargs.get('extinction_law', ext.fitzpatrick99)


        


    def _initialise_filter_names(self,filters,use_isochrones):
        '''
        Get the filters from the input
        '''
        isochrone_filters=None
        if(filters=='default'):
            print('Setting default filters')
            dust_filters=['GAIA/GAIA3.G','GAIA/GAIA3.Gbp', 'GAIA/GAIA3.Grp','2MASS/2MASS.J', '2MASS/2MASS.H', '2MASS/2MASS.Ks','WISE/WISE.W1','WISE/WISE.W2']
        else:
            dust_filters=filters
        if(use_isochrones):
            name_mapping = {
            'GAIA/GAIA3.G': 'Gaia_G_EDR3',
            'GAIA/GAIA3.Gbp': 'Gaia_BP_EDR3',
            'GAIA/GAIA3.Grp': 'Gaia_RP_EDR3',
            '2MASS/2MASS.J': '2MASS_J',
            '2MASS/2MASS.H': '2MASS_H',
            '2MASS/2MASS.Ks': '2MASS_Ks',
            'WISE/WISE.W1': 'WISE_W1',
            'WISE/WISE.W2': 'WISE_W2'
            }
            try:
                isochrone_filters=[name_mapping[filt] for filt in dust_filters]
            except:
                isochrone_filters=None
                print('At least one of the filters isnt in MIST')      
        return dust_filters,isochrone_filters
    def _initialise_transmission_functions(self):
        '''
        This assumes constant passbands.
        '''
        if(self._load_filter_names()):
            try:
                with open(self.loc_passbands, 'rb') as file:
                    data = pickle.load(file)
                    return data
            except FileNotFoundError:
                    print(f"File not found: {self.loc_passbands}")
                    print('Creating pickle file')
                    self._generate_passbands()
                    return self._initialise_transmission_functions()
        else:
            self._save_filter_names()
            print('Different name passband')
            print('Creating pickle file')
            self._generate_passbands()
            return self._initialise_transmission_functions()

    def _generate_passbands(self):
        try:
            passbands = svo.get_svo_passbands(self.dustapprox_filters)
        except:
            self.dustapprox_filters,self.isochrone_filters=self._initialise_filter_names(filters=None,use_isochrones=self.use_isochrones)
            passbands = svo.get_svo_passbands(self.dustapprox_filters)

        with open(self.loc_passbands, 'wb') as file:
            pickle.dump(passbands, file)

    def _save_filter_names(self):
        with open(self.passband_names_location, 'w') as file:
            for filter_name in self.dustapprox_filters:
                file.write(filter_name + '\n')

    def _load_filter_names(self):
        try:
            with open(self.passband_names_location, 'r') as file:
                loaded_filters = [line.strip() for line in file]
        except:
            return False

        # Check if loaded filters match expected filters
        if loaded_filters == self.dustapprox_filters:
            return True
        else:
            return False

    def _initialise_interpolator(self):
        '''
        This assumes constant passbands.
        '''
        if(self._load_interpolation_parameters()):
            try:
                print('Pickle interpolator file exists')
                with open(self.interpolator_location, 'rb') as file:
                    interpolator = pickle.load(file)
                with open(self.lamb_unit_location, 'rb') as file:
                    lamb_unit = pickle.load(file)
                with open(self.flux_unit_location, 'rb') as file:
                    flux_unit = pickle.load(file)
                wavelength=np.load(self.wavelength_location)
                return interpolator,lamb_unit,flux_unit,wavelength
            except FileNotFoundError:
                    print(f"File not found: {self.interpolator_location}")
                    print('Creating pickle file')
                    interpolator,lamb_unit,flux_unit,wavelength=self._interpolator_choice(atmospheres=self.interpolation_details[0],how=self.interpolation_details[1])
                    with open(self.interpolator_location, 'wb') as file:
                        pickle.dump(interpolator, file)
                    with open(self.lamb_unit_location, 'wb') as file:
                        pickle.dump(lamb_unit, file)
                    with open(self.flux_unit_location, 'wb') as file:
                        pickle.dump(flux_unit, file)   
                    np.save(self.wavelength_location,wavelength)                    
                    return self._initialise_interpolator()
        else:
            self._save_interpolation_parameters()
            print('Different combination parameters')
            print('Creating interpolator pickle file')
            interpolator,lamb_unit,flux_unit,wavelength=self._interpolator_choice(atmospheres=self.interpolation_details[0],how=self.interpolation_details[1])
            with open(self.interpolator_location, 'wb') as file:
                pickle.dump(interpolator, file)
            with open(self.lamb_unit_location, 'wb') as file:
                pickle.dump(lamb_unit, file)
            with open(self.flux_unit_location, 'wb') as file:
                pickle.dump(flux_unit, file)         
            np.save(self.wavelength_location,wavelength)  
            return self._initialise_interpolator()

    def _interpolator_choice(self,atmospheres,how):
        if(atmospheres=='ATLAS9' and how=='LINEAR'):
            return atmosphere_interpolators.atlas_linear()
        else:
            raise NotImplementedError


    def _save_interpolation_parameters(self):
        with open(self.atmosphere_names_location, 'w') as file:
            for param in self.interpolation_details:
                file.write(param + '\n')

    def _load_interpolation_parameters(self):
        try:
            with open(self.atmosphere_names_location, 'r') as file:
                loaded_interpolator = [line.strip() for line in file]
        except:
            return False

        # Check if loaded filters match expected filters
        if loaded_interpolator == self.interpolation_details:
            return True
        else:
            return False

class SEDNest_ISO(SEDNest):
    def __init__(self, *args, **kwargs):
        '''
        Create a subclass of the SED Nested sampling fitter. This is for using Isochrones as direct
        priors in the forward modelling. Default is MIST using the minimint package.
        
        -->kwargs
        isochrone_type: system of isochrones to use. 


        -->
        self.isochrone_interpolator -- returns the function to interpolate over the parameters

        '''
        super().__init__(*args, **kwargs)

        self.isochrones_avail=['mist']
        self.isochrone_type = kwargs.get('isochrone_type', None)
        if isinstance(self.isochrone_type, str) and self.isochrone_type == "mist":
            pass
        else:
            warning_message = "Only available isochrone keywords are: isochrone_type=" + ', '.join(self.isochrones_avail)
            warnings.warn(warning_message, UserWarning)
            warnings.warn('Defaulting to mist', UserWarning)
            self.isochrone_type = "mist"

        self.isochrone_interpolator=self._interpolator_type()
        self.params=self._get_param_names()

    def _get_param_names(self):
        # returns the interpolator for the different isochrones
        if(self.isochrone_type=='mist'):
            return ['mass','feh','age','extinction','distance','teff','Rv']
        else:
            raise NotImplementedError
    
    def _interpolator_type(self):
        # returns the interpolator for the different isochrones
        if(self.isochrone_type=='mist'):
            return minimint.Interpolator(self.filters,data_prefix=self.data_prefix)
        else:
            raise NotImplementedError


    def MIST_Interpolator(self,mass,logage,feh):
        ### Uses the Minimint Package to Interpolate over Consistent Stellar Parameters
        if(self.isochrone_type=='mist'):
            stellar_model_parameters=self.isochrone_interpolator(mass,logage,feh)
        else:
            raise NotImplementedError
        return stellar_model_parameters





print(SEDNest(use_isochrones=True).interpolate_spectra(feh=np.array([0,-1]),teff=np.array([5000,7000]),logg=np.array([4.5,4.7])).shape)