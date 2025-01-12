import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from astropy.io import fits
from astropy.table import Table
import emcee
from scipy.interpolate import Akima1DInterpolator
from scipy.optimize import curve_fit
import seaborn as sb

#plt.style.use('ggplot')
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 15

def gaussian(x, A, mu, sigma):
    
    '''
    Gaussian Model for Line Fitting. This is of the form:
    
    Gaussian = Ae^(-(x-mu)^2/sigma^2)
    
    Input
    -------------
    x: array or single value to evaluate the Gaussian 
    A: amplitude of the Gaussian
    mu: Center of the Gaussian
    sigma: the standard deviation of the Gaussian
    
    
    Returns
    --------------
    Evaluated Gaussian for the given A, mu and sigma at the point(s) in x
    
    '''
    
    return A * np.exp(-(x - mu)**2/ (sigma**2))

def line(x, m, b):
    
    '''
    Continuum of the spectra using y = b
    
    Input
    ------------
    x: array of values
    b: value to plot y = b
    
    
    Returns
    ------------
    An array of values at b, the same size as x
    
    '''
    
    return  m*x + b

def line_model(x, A, mu, sigma, m, b):
    
    '''
    Emission Line model using Gaussian and the continuum
    
    Inputs
    ------------

    x: array of values to evaluate the Gaussian 
    A: amplitude of the Gaussian
    mu: Center of the Gaussian
    sigma: the standard deviation of the Gaussian
    b: value to plot y = b
    '''
    
    
    return gaussian(x, A, mu, sigma) + line(x, m, b)

def log_likelihood(theta, x, y, yerr):
    '''
    This is the likelihood function we are using for emcee to run
    
    This likelihood function is the maximum likelihood assuming gaussian errors.
    
    '''
    ################
    
    # The value we are trying to fit
    #A, mu, sigma, m, b = theta
    
    #Making the model of the emission line
    model = line_model(x, *theta)
    
    #getting the log likelihood, this is similar to chi2 = sum((data - model)^2/sigma^2)
    lnL = -0.5 * np.sum((y - model) ** 2 / yerr**2)
    
    return lnL


def log_prior(theta, wave_center, Amp_max):
    '''
    The prior function to be used against the parameters to impose certain criteria for the fitting making 
    sure that they do not go and explore weird values
    
    '''
    #Theta values that goes into our Gaussian Model
    A, mu, sigma, m, b = theta
    
    #the left most bound and right most bound that the central wavelength can vary
    left_mu = wave_center - .02  # this is how much mu can vary
    right_mu = wave_center + .02 # this is how much mu can vary
    
    #min and max amplitude of the emission line
    min_A = 0
    max_A = Amp_max * 2
    
    sigma_window_left = .01 #had to change these for the input spectra these are left bounds for sigma
    sigma_window_right = .05 #had to change these for the input spectra these are right bounds for sigma
        
    if (0 < A < max_A) & (left_mu <= mu <= right_mu) & (sigma_window_left <= sigma < sigma_window_right):
        return 0.0
    else:
        return -np.inf

def log_probability(theta, x, y, yerr, first_wave, Amp_max):
    
    lp = log_prior(theta, first_wave, Amp_max)
    if not np.isfinite(lp):
        #print('Probability is infinite')
        return -np.inf
    prob = lp + log_likelihood(theta, x, y, yerr)
    #print(f'Prob:{prob:.3E}')
    return prob


def initial_fits(wave, spectrum, err_spec, window, line_center, diagnose = False):
    
    '''
    This function does an initial fit on the data using curve fit which we then pass in those parameters into emcee
    to do the full MCMC fit later
    
    Inputs
    -------------
    wave: Wavelength Array
    spectrum: Full spectrum array
    err_spec: the Error spectra
    window: the window to look around an emission line in units of the wavelength array
    line_center: The line center of the emission line
    
    Returns:
    
    result: The output of the intial curve fit which would be an array with output in order of the parameters
            in the model np.array([A, mu, sigma, b])
    
    '''
    
    #the range where the optimization can look between 
    min_window = line_center - window
    max_window = line_center + window
    
    #getting emission line near the line center
    #line_center +/- window
    indx = np.where((min_window < wave) & ((wave < max_window)))[0]

    spec_window = spectrum[indx]
    wave_window = wave[indx]
    err_spec_window = err_spec[indx]
    
    #initial guesses for the optimization
    guess_A = np.amax(spectrum[indx])
    guess_mu = line_center
    
    #We interpolate the spectrum near the emission line, we do this to get an estimate on sigma by computing 
    #the full width at half-maximum
    spec_interp = Akima1DInterpolator(wave_window, spec_window)
    
    #making a wavelength array near the emission line
    x = np.linspace(wave_window[0], wave_window[-1], 10000)
    
    #applying the wavelength array to the interpolated function
    spec = spec_interp(x)
    
    #getting the value at half maximum
    half_max = np.amax(spec)/2
    
    #finding index where the spectrum is higher than the half-maximum value
    #the first and last indexes are the wavelength where the sigma can be computed
    idx = np.where(spec > half_max)[0]
    
    #getting the left and right most wavelengths
    wave_left, wave_right = x[idx[0]], x[idx[-1]]
    
    #taking the difference between the right and left wavelength and divide it by 2 to get a guess for the sigma
    guess_sigma = (wave_right - wave_left)/2

    x = [wave_window[0], wave_window[-1]]
    y = [np.median(wave_window[:5]), [np.median(wave_window-5:])]]
    
    m_guess, b_guess = np.polyfit(x, y, 1)
    
    if diagnose == True:
        
        print('Minimization Guesses')
        print(f"A: {guess_A}")
        print(f"mu: {guess_mu}")
        print(f"sigma: {guess_sigma}")
        print(f"b: {np.median(spec_window)}")
        print() 

    #making initial guesses
    x0 = [guess_A, guess_mu, guess_sigma, m_guess, b_guess]

    #making lower and upper bounds to use into curve_fit
    low_bounds = [0, min_window, 0, -np.inf, np.median(spec_window)/2]
    high_bounds = [2*guess_A, max_window, .1, np.inf, np.median(spec_window)*2]
    
    # Optimization of the initial gaussian fit
    result,_ = curve_fit(line_model, wave_window, spec_window, p0 = x0, 
                          bounds = [low_bounds, high_bounds])                 
    
    
    ########
    # Diagnostic Plotting: making sure we are getting the emission line
    ########
    if diagnose == True:
        
        print('Minimization Results')
        print(f"A: {result[0]}")
        print(f"mu: {result[1]}")
        print(f"sigma: {result[2]}")
        print(f"b: {result[3]}")
        print()
        
        xarr = np.linspace(wave_window[0], wave_window[-1], 100)
        plt.figure()
        plt.plot(wave_window, spec_window, color = 'blue', label = 'Data')
        plt.scatter(wave_window, spec_window, color = 'blue')
        plt.plot(xarr, line_model(xarr, *result), color = 'black', label = 'Model')
        plt.axhline(0, linestyle = '--')
        plt.ylabel('Flux')
        plt.xlabel(r'Wavelength $\mu$m')
        plt.title('Initial curve_fit Fitting')
        plt.legend()
        plt.show()
    
    
    return result

def fitting_line(wave, flux, flux_err, line_center, window_wavelength, run = 3000,
                 diagnose = False,save_df=True, save_spec = False, 
                 file_spec = 'Emcee_Spectra.txt', 
                 filename = 'Emcee_Chains_Galaxy.txt'):
    
    '''
    The code that fits the line using the emcee approach
    
    Inputs
    -----------
    wave: Wavelength array
    flux: Flux array
    flux_err: Flux error array
    line_center: the line center
    window_wavelength: The window near emission line in units of wavelength
    run: how many iterations to run emcee on default is 3000
    diagnose: An optional argument to output diagnostic plots as the fitting is proceeding
              Outputs plots from the initial fits, walker locations prior to using emcee and output emcee plots
    save_df: Saving the emcee df output to a file
    save_spec: Saving spectra used in emcee fitting to a file
    file_spec: name of the file for the emcee spectra
    filename:The name of the file to save the emcee output
    
    Returns
    -----------
    emcee_wave: The wavelength array used in the emcee fitting 
    emcee_spec: the flux spectra array used in the emcee fitting
    emcee_err: The error array used in the emcee fitting 
    emcee_df: the output emcee data frame with parameter values and flux estimates using Flux = A*sigma*sqrt(2 pi)
    '''
    
    
    #calling the function that does the initial fitting
    result = initial_fits(wave, flux, flux_err, window_wavelength, line_center, diagnose = diagnose)
    
    #getting the results from the initial fit to then pass into emcee
    guess_A = result[0]
    guess_mu = result[1]
    guess_sigma = result[2]
    guess_m = result[3]
    guess_b = result[4]
    
    
    #making walkers so that we can use emcee to explore the parameter space
    #centered on the best results from minimization
    amp_jump = np.random.normal(loc = guess_A,            #centered on best A from curve_fit
                                scale = guess_A/10,       #can wander 1/10 of the value of A
                                size = 32).reshape(-1, 1) 
    
    wavelength_jump = np.random.normal(loc = guess_mu,    #centered on best mu from curve_fit
                                       scale = .005,      #can wander +/- 0.005 microns (again tailored to nirspec
                                       size = 32).reshape(-1, 1)#data so if you are working with other spectra 
                                                                #you may need ot update this)
    
    sigma_jump = np.random.normal(loc = guess_sigma,       #centered on best sigma from curve_fit
                                  scale = .02,            #can wander +/- 0.002 microns (tailored for nirspec data)
                                  size = 32).reshape(-1, 1)

    powerb = np.log10(np.abs(guess_m))
    
    m_jump = np.random.normal(loc = guess_m,       #centered on best sigma from curve_fit
                                  scale = 1*10**powerb,        
                                  size = 32).reshape(-1, 1)

    
    #getting the power of 10 that the linear fit is
    powerb = np.log10(np.abs(guess_b))
    
    #
    b_jump = np.random.normal(loc = guess_b,           #centered on best b from curve_fit
                              scale = 1*10**powerb,    #making it wander 10^powerb (if b = .05, it can wander .01)
                              size = 32).reshape(-1, 1)

    
    #################
    # Diagnostic plotting to see if the parameters were jumping to large values
    # The should be concentrated near their best fit results values
    #################
    if diagnose == True:
        print('Checking the Walker Jumps')
        fig, ax = plt.subplots(nrows = 2, ncols = 3, constrained_layout = True)
        
        ax[0, 0].hist(amp_jump)
        ax[0, 0].set_xlabel('Amplitude')
        
        ax[0, 1].hist(wavelength_jump)
        ax[0, 1].set_xlabel(r'$\mu$')
        
        ax[1, 0].hist(sigma_jump)
        ax[1, 0].set_xlabel(r'$\sigma$')

        ax[1, 1].hist(m_jump)
        ax[1, 1].set_xlabel(r'$\sigma$')
        
        ax[2, 0].hist(b_jump)
        ax[2, 0].set_xlabel('b')
        
        plt.show()
    

    #stacking along the columns and generating the starting walkers
    starting_walkers = np.hstack((amp_jump,
                                  wavelength_jump, 
                                  sigma_jump,
                                  m_jump,
                                  b_jump))

    #initializing window for emcee around the best result mu
    emcee_window = window_wavelength 
    
    #getting indexes near the emission line based off of the emcee_window
    #looking at line_center +/- emcee_window
    emcee_indx = np.where((wave >= (line_center - emcee_window)) & 
                          (wave <= (line_center + emcee_window)))[0] 

    #emcee subsections
    emcee_spec = flux[emcee_indx]
    emcee_wave = wave[emcee_indx]
    emcee_err = flux_err[emcee_indx]
    
    
    ###########
    #NOTE:
    #need to change output name everytime you run otherwise it will overwrite
    ###########
    
    # if save_spec == True:
    #     #saves the input emcee spectra
    #     emcee_spec_matrix = np.c_[emcee_wave, emcee_spec, emcee_err]
    
    #     np.savetxt(file_spec, emcee_spec_matrix)

    #initializing walker positions
    pos = starting_walkers
    nwalkers, ndim = pos.shape

    #initializing sampler
    sampler = emcee.EnsembleSampler(nwalkers, #giving emcee the walker positions
                                    ndim,     #giving it the dimension of the model(same as number of model parameters)
                                    log_probability, #giving it the log_probability function
                                    args=(emcee_wave, emcee_spec, emcee_err, guess_mu, guess_A), #arguments to pass into log_probability
                                    moves = [(emcee.moves.DEMove(), 0.5),        
                                             (emcee.moves.DESnookerMove(), 0.5)])

    #running emcee
    state = sampler.run_mcmc(pos, 1000)
    sampler.reset()
    sampler.run_mcmc(state, run, progress=False)

    #getting values back
    flat_samples = sampler.get_chain(flat=True)
    LnL_chain = sampler.flatlnprobability
    
    emcee_df = pd.DataFrame()
    emcee_df['A'] = flat_samples[:, 0]
    emcee_df['mu'] = flat_samples[:, 1]
    emcee_df['sigma'] = flat_samples[:, 2]
    emcee_df['m'] = flat_samples[:, 3]
    emcee_df['b'] = flat_samples[:, 4]
    emcee_df['LnL'] = LnL_chain[:]
    
    #removing values where the log_likelihood was infinite as these are bad fits
    emcee_df = emcee_df[np.isfinite(emcee_df.LnL.values)]
    
    #getting the flux from the parameter values
    fluxes_emcee = emcee_df['A'] * emcee_df['sigma'] * np.sqrt(2 * np.pi)
    
    emcee_df['Fluxes'] = fluxes_emcee
    
    if diagnose == True:
        
        print('Checking Prameter Posterior Distributions')
        fig, ax = plt.subplots(nrows = 3, ncols = 2, constrained_layout = True)
        
        emcee_df.A.hist(ax = ax[0, 0])
        emcee_df.mu.hist(ax = ax[0, 1])
        emcee_df.sigma.hist(ax = ax[1, 0])
        emcee_df.m.hist(ax = ax[1, 1])
        #emcee_df.m.hist(ax = ax[1, 0])
        emcee_df.b.hist(ax = ax[2, 0])
        
        plt.show()
    
    if diagnose == True:
        xarr = np.linspace(emcee_wave[0], emcee_wave[-1], 100)
        
        plt.figure()
        plt.title('Input Emcee Spectra and Emcee Fit')
        plt.plot(emcee_wave, emcee_spec, color = 'black', alpha = 0.5, label = 'Data')
        plt.scatter(emcee_wave, emcee_spec, color = 'black')
        plt.plot(xarr, line_model(xarr, *emcee_df.quantile(q = 0.5).values[:-2]), label = 'Model')
        plt.xlabel(r'Wavelength [$\mu$m]')
        plt.ylabel('Flux')
        plt.legend()
        plt.show()
    
    ###########
    #NOTE:
    #need to also give the filename argument otherwise it will overwrite the default file
    ###########
    if save_df == True:
        emcee_df.to_csv(filename, sep = ' ')
        
    else:
        return emcee_wave, emcee_spec, emcee_err, emcee_df