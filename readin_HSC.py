from astropy.io import fits
import pandas as pd
import numpy as np
from astropy.table import Table, vstack

# add Smearing                                                                                                             
addSmearing = False

def df_colors(df, mag_flag):

    df['g_r']=(df['g']-df['r'])
    df['r_z']=(df['r']-df['z'])
    df['g_z']=(df['g']-df['z'])

    df['g_W1']=(df['g']-df['W1'])
    df['r_W1']=(df['r']-df['W1'])
    df['z_W1']=(df['z']-df['W1'])
    df['g_W2']=(df['g']-df['W2'])
    df['r_W2']=(df['r']-df['W2'])
    df['z_W2']=(df['z']-df['W2'])
    df['W1_W2']=(df['W1']-df['W2'])

    df.keep_columns(['g_r','r_z','g_z','g_W1','r_W1','z_W1','g_W2','r_W2','z_W2','W1_W2','r','g','z','W1','W2','zspec'])
    return df


def magsExtFromFlux(df):

    gflux  = df['DECAM_FLUX'][:,1]/df['DECAM_MW_TRANSMISSION'][:,1]
    rflux  = df['DECAM_FLUX'][:,2]/df['DECAM_MW_TRANSMISSION'][:,2]
    zflux  = df['DECAM_FLUX'][:,4]/df['DECAM_MW_TRANSMISSION'][:,4]
    W1flux = df['WISE_FLUX'][:,0]/df['WISE_MW_TRANSMISSION'][:,0]
    W2flux = df['WISE_FLUX'][:,1]/df['WISE_MW_TRANSMISSION'][:,1]
    zspec = df['zspec']
    W1flux[np.isnan(W1flux)]=0.
    W2flux[np.isnan(W2flux)]=0.
    gflux[np.isnan(gflux)]=0.
    rflux[np.isnan(rflux)]=0.
    zflux[np.isnan(zflux)]=0.
    W1flux[np.isinf(W1flux)]=0.
    W2flux[np.isinf(W2flux)]=0.
    gflux[np.isinf(gflux)]=0.
    rflux[np.isinf(rflux)]=0.
    zflux[np.isinf(zflux)]=0.

    g=np.where( gflux>0,22.5-2.5*np.log10(gflux), 0.)
    r=np.where( rflux>0,22.5-2.5*np.log10(rflux), 0.)
    z=np.where( zflux>0,22.5-2.5*np.log10(zflux), 0.)
    W1=np.where( W1flux>0, 22.5-2.5*np.log10(W1flux), 0.)
    W2=np.where( W2flux>0, 22.5-2.5*np.log10(W2flux), 0.)
    
    g[np.isnan(g)]=0.
    g[np.isinf(g)]=0.
    r[np.isnan(r)]=0.
    r[np.isinf(r)]=0.
    z[np.isnan(z)]=0.
    z[np.isinf(z)]=0.
    W1[np.isnan(W1)]=0.
    W1[np.isinf(W1)]=0.
    W2[np.isnan(W2)]=0.
    W2[np.isinf(W2)]=0.

    df['r']=r
    df['g']=g
    df['z']=z
    df['W1']=W1
    df['W2']=W2
    return df

def bandSmearing(df):

# 5 sigma depths                                                                                                           
    limitInf=1.e-04
    depth_g = 24.0
    depth_r = 23.4
    depth_z = 22.5
    nbElements=len(df)
    np.random.seed(0)
    s_g=np.random.normal(0.,1,nbElements)
    s_r=np.random.normal(0.,1,nbElements)
    s_z=np.random.normal(0.,1,nbElements)

    gflux = np.power(10.,0.4*(22.5-df['g'])) + 1./5.*s_g*np.power(10.,0.4*(22.5-depth_g))
    rflux = np.power(10.,0.4*(22.5-df['r'])) + 1./5.*s_r*np.power(10.,0.4*(22.5-depth_r))
    zflux = np.power(10.,0.4*(22.5-df['z'])) + 1./5.*s_z*np.power(10.,0.4*(22.5-depth_z))

    df['g'] = np.where( gflux>limitInf,22.5-2.5*np.log10(gflux), 0.)
    df['r'] = np.where( rflux>limitInf,22.5-2.5*np.log10(rflux), 0.)
    df['z'] = np.where( zflux>limitInf,22.5-2.5*np.log10(zflux), 0.)

    return df

def colorSelection(df):

    df = df[ (df['r']<23.4) & ((df['r']-df['z'])>0.3) & ((df['r']-df['z'])<1.6) &
           ((df['g']-df['r'])< (1.15*(df['r']-df['z'])-0.15)) & 
           ( (df['g']-df['r']) < (1.6-1.2*(df['r']-df['z'])) )]
    return df

# Z photometric                                     
def sort_data(mag_flag):
    dataDir='/global/homes/y/yeche/ELG_BDT/data-decals/'
    dataLOWTraining = dataDir+'HSC_Train.fits' #dr3     
    ELGTraining     = dataDir+'HSC_Train.fits' #dr3 
    dataELG_o = fits.open(ELGTraining)[1].data
    dataELG=Table(dataELG_o)
    del dataELG_o
    dataELG = dataELG[~np.isnan(dataELG['zspec'])]
    dataELG = magsExtFromFlux(dataELG)
    if (mag_flag ==1):
        dataELG= dataELG[(dataELG['g']>0)&(dataELG['r']>0)&(dataELG['z']>0)&(dataELG['g']<23.6)
                   &(dataELG['zspec']>0.6)&(dataELG['zspec']<1.6)]
    if (mag_flag ==2):
        dataELG= dataELG[(dataELG['g']>0)&(dataELG['r']>0)&(dataELG['z']>0)&(dataELG['r']<23.6)
                   &(dataELG['zspec']>0.6)&(dataELG['zspec']<1.6)]

    dataELG = df_colors(dataELG,mag_flag)
    dataLOW_o = fits.open(dataLOWTraining)[1].data
    dataLOW=Table(dataLOW_o)
    del dataLOW_o
    dataLOW = dataLOW[~np.isnan(dataLOW['zspec'])]
    dataLOW = magsExtFromFlux(dataLOW)
   
    dataLOW = dataLOW[(dataLOW['g']>0)&(dataLOW['r']>0)&(dataLOW['z']>0)&(dataLOW['r']<23.6)
              &(dataLOW['zspec']>0.1)&(dataLOW['zspec']<0.6)]

    dataLOW = df_colors(dataLOW,mag_flag)

    print("assign cat")
    dataELG['y']=1
    dataLOW['y']=0
    alldata =(vstack([dataELG, dataLOW]))
    #alldata = pd.concat([dataELG, dataLOW], ignore_index=True)
    shuffled_data =Table(np.random.permutation(alldata))
    return shuffled_data
    
def sort_test_data(mag_flag):
    testDir='/global/homes/y/yeche/ELG_BDT/data-decals/'
    testdata = testDir+'HSC_Test.fits' #dr3 
    data_o = fits.open(testdata)[1].data[300000:450000]
    dataTest=Table(data_o)
    del data_o
    dataTest = dataTest[~np.isnan(dataTest['zspec'])]
    dataTest = magsExtFromFlux(dataTest)
    dataTest = df_colors(dataTest,mag_flag)
    if (addSmearing)  :
        dataTest = bandSmearing(dataTest)
        dataTest = colorSelection(dataTest)
        dataTest = df_colors(dataTest, mag_flag)
    dataTest['y'] = 0
    dataTest['y'][np.where(dataTest['zspec'] > 0.6)] = 1

    return dataTest


