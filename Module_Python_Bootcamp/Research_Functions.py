from imports import *

def reading_file(filename):
    
    '''
    Function that reads in the data
    
    Input
    -------
    filename: the filename or filepath of the file to read must be in csv format
    
    
    Returns
    -----------
    data: data in the form of a pandas dataframe
    '''
    
    data = pd.read_csv(filename)
    
    return data

def filter_data(DF):
    
    '''
    Function to filter the data according to RA and DEC
    
    Input
    -----------
    DF: data (pandas Dataframe)
    
    Returns
    -----------
    filtered_DF
    '''
    
    RA_region = (270 < DF.RA.values) & (DF.RA.values < 271)
    DEC_region = (70 < DF.DEC.values) & (DF.DEC.values < 71)
    
    mask = RA_region & DEC_region
    filtered_DF = DF[mask]
    
    return filtered_DF

def high_z_selector(DF):
    
    '''
    Function to select high-z sources
    
    Inputs
    -----------
    
    DF: DF after it has been filtered
    
    Returns
    ------------
    
    DF 
    '''
    
    high_z_mask = DF.zphot > 6
    
    return DF[high_z_mask]