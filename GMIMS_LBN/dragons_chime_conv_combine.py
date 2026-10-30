import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from scipy import interpolate
from reproject import reproject_interp
import copy
from tqdm.auto import tqdm
from scipy.stats import linregress
from astropy.coordinates import SkyCoord
import healpy as hp
import reproject
from scipy import interpolate

def run_all():

    print('doing this in steps for now')

    return

### The main programs: #####

# (1) # Make FITS files of CHIME in Jy/beam, DRAGONS in K on matched grids (spatial and frequency)
def regrid_and_dragonsK(chime_file, dragons_file, dragons_f_file, out_dir, stokes):

    print('')
    print('=====================================')
    print('Matching CHIME, DRAGONS gridding...')
    print('=====================================')
    print('')

    chime, dragons, wcs_chime, wcs_dragons = read_data(chime_file, dragons_file)
    print(chime.shape)
    print(dragons.shape)

    freq_chime, freq_dragons = get_freqs(wcs_chime, wcs_dragons, freqlist1=None, freqlist2=dragons_f_file)
    print(freq_chime.shape)
    print(freq_dragons.shape)
    
    dragons_K = dragons_to_K(dragons,freq_dragons, out_dir)
    dragons_K = dragons_nans(dragons_K, wcs_dragons)
    dragons_K_regrid = regrid(chime, dragons_K, wcs_chime, wcs_dragons, out_dir)
    print(dragons_K_regrid.shape)

    chime_match, dragons_match, freqs_new = match_freqs(chime,dragons_K_regrid,freq_chime,freq_dragons,out_dir)
    make_fits_file_eq(chime_match, freqs_new, out_dir, 'chime_test_'+stokes+'.fits')
    make_fits_file_eq(dragons_match, freqs_new, out_dir, 'dragons_test_'+stokes+'.fits')

    return 

# (2) # Make FITS files of CHIME and DRAGONS with spatial filtering
def band_pass(chime_file, dragons_file, out_dir, stokes, Rlims, filt_version):

    print('')
    print('========================')
    print('Filtering datasets...')
    print('========================')
    print('')

    chime, dragons, wcs_chime, wcs_dragons = read_data(chime_file, dragons_file)
    freq_chime, freq_dragons = get_freqs(wcs_chime, wcs_dragons, freqlist1=None, freqlist2=None)

    #chime, dragons = mask_bright_srcs(chime, dragons, wcs_chime, rmax=2)
    #chime, dragons = mask_bad_RA(chime, dragons, wcs_chime)

    if np.array_equal(freq_chime, freq_dragons):

        chime_filt   = np.empty_like(chime)
        dragons_filt = np.empty_like(dragons)

        for idx in tqdm(range(0,len(freq_chime))):

            #image_FFT_shift, mask, SA_beam, xuv, yuv, dragons_slice = SA_deconvolve(dragons[idx], 5.0, 0.125, freq_dragons[idx])

            chime_filt_slice,   mask, uv_m, uv_n = band_pass_slice(chime[idx], 0.125, freq_chime[idx], 
                                                            R1min=Rlims[0], R1max=Rlims[1], R2min=Rlims[2], R2max=Rlims[3])
            dragons_filt_slice, mask, uv_m, uv_n = band_pass_slice(dragons[idx], 0.125, freq_dragons[idx], 
                                                            R1min=Rlims[0], R1max=Rlims[1], R2min=Rlims[2], R2max=Rlims[3])
            chime_filt[idx]   = chime_filt_slice.real
            dragons_filt[idx] = dragons_filt_slice.real

        make_fits_file_eq(chime_filt, freq_chime, out_dir, 'chime_filtered_'+stokes+'_'+filt_version+'.fits')
        make_fits_file_eq(dragons_filt, freq_dragons, out_dir, 'dragons_filtered_'+stokes+'_'+filt_version+'.fits') 

    else:
        print('Frequencies do not match! Stopping here.')

    return

# (3) # Get the conversion factors for CHIME into K
def compare_chime_dragons(chime_file, dragons_file, out_dir, stokes):

    print('')
    print('=================================')
    print('Calculating conversion factors...')
    print('=================================')
    print('')

    chime, dragons, wcs_chime, wcs_dragons = read_data(chime_file, dragons_file)
    freq_chime, freq_dragons = get_freqs(wcs_chime, wcs_dragons, freqlist1=None, freqlist2=None)

    if np.array_equal(freq_chime, freq_dragons):

        chime, dragons = mask_bright_srcs(chime, dragons, wcs_chime, rmax=2)
        chime, dragons = mask_bad_RA(chime, dragons, wcs_chime)
        chime, dragons = mask_bright_regions(chime, dragons)

        #fig,ax = plt.subplots(1,2,figsize=(40,12))
        #idx = 0
        #ax[0].imshow(chime[idx,:,:],origin='lower',vmin=-1,vmax=1,cmap='RdBu_r')
        #ax[1].imshow(dragons[idx,:,:],origin='lower',vmin=-1,vmax=1,cmap='RdBu_r')
        #plt.savefig(out_dir+'masking_srcs_'+stokes+'.pdf',transparent=False)

        #dec_chime_vs_dragons(chime, dragons, wcs_chime, freq_chime, stokes, out_dir, 
        #                     ddec=1, makeplots=True)

        dec_chime_vs_dragons_hpx(chime, dragons, wcs_chime, freq_chime, stokes, out_dir, 
                                 nside=64, dec_min=14, makeplots=True)

        #make_fits_file_eq(chime, freq_chime, out_dir, 'chime_test_K'+stokes+'.fits')

    else:
        print('Frequencies do not match! Stopping here.')

    
    return

# (4) # Make FITS files of CHIME with conversion factor applied
def chimeK(chime_file, conversion_file, out_dir, stokes):

    print('')
    print('=================================')
    print('Applying K conversion factor to CHIME...')
    print('=================================')
    print('')
        
    chime = fits.open(chime_file)[0].data
    wcs = WCS(fits.open(chime_file)[0].header)

    #params = np.load(conversion_file)
    #print(params['decs'].shape)
    #print(params['slopes'].shape)
    #print(params['freqs'].shape)
    #print(chime.shape)

    freq_chime = wcs.all_pix2world(0,0,range(wcs.array_shape[0]),0)[2]/1e6
    decs_chime  = wcs.all_pix2world(0,range(wcs.array_shape[1]),0,0)[1]
    #print(decs_chime)

    slopes_smooth, ints_smooth = smooth_conversion(conversion_file, out_dir, stokes, decs_chime)

    chime_new = np.expand_dims(slopes_smooth,axis=2)*chime + np.expand_dims(ints_smooth,axis=2)
    make_fits_file_eq(chime_new, freq_chime, out_dir, 'chime_K_'+stokes+'.fits')


    #if ((np.array_equal(freq_chime, params['freqs'])) & (np.array_equal(decs_chime, params['decs']))):

    #    chime_new = np.expand_dims(params['slopes'],axis=2)*chime + np.expand_dims(params['ints'],axis=2)
    #    make_fits_file_eq(chime_new, freq_chime, out_dir, 'chime_K_'+stokes+'.fits')

    #else:
    #    print('Frequencies and/or declinations do not match! Stopping here.')

    
    return

    
# (5) # Make FITS file of combined CHIME+DRAGONS, along with intermediate outputs
def combine(chime_file, dragons_file, out_dir, stokes, Rlims, filt_version):

    print('')
    print('=============================')
    print('Combining CHIME and DRAGONS')
    print('=============================')
    print('')

    chime, dragons, wcs_chime, wcs_dragons = read_data(chime_file, dragons_file)

    freq_chime, freq_dragons = get_freqs(wcs_chime, wcs_dragons, freqlist1=None, freqlist2=None)

    if np.array_equal(freq_chime, freq_dragons):

        chime_feath   = np.empty_like(chime)
        dragons_feath = np.empty_like(dragons)
        dragons_deconv = np.empty_like(dragons)

        for idx in tqdm(range(0,len(freq_chime))):

            image_FFT_shift, mask, SA_beam, xuv, yuv, dragons_slice = SA_deconvolve(dragons[idx], 5.0, 0.125, freq_dragons[idx])
            dragons_deconv[idx] = dragons_slice

            chime_feath[idx], mask, uv_m, uv_n   = feather(chime[idx], 0.125, Rlims, 'AS', freq_chime[idx])
            dragons_feath[idx], mask, uv_m, uv_n = feather(dragons_deconv[idx], 0.125, Rlims, 'SA', freq_dragons[idx])

        make_fits_file_eq(dragons_deconv, freq_dragons, out_dir, 'dragons_deconv_'+stokes+'_'+filt_version+'.fits')
        make_fits_file_eq(dragons_feath,  freq_dragons, out_dir, 'dragons_feath_'+stokes+'_'+filt_version+'.fits') 
        make_fits_file_eq(chime_feath,    freq_chime,   out_dir, 'chime_feath_'+stokes+'_'+filt_version+'.fits')

        make_fits_file_eq(chime_feath+dragons_feath, freq_chime, out_dir, 'chime_dragons_'+stokes+'_'+filt_version+'.fits')  
            

    else:
        print('Frequencies do not match! Stopping here.')

    return



##########################
### The subroutines: #####
##########################


def read_data(file1, file2):

    print('')
    print('Reading in data...')
    print('')
        
    data1 = fits.open(file1)[0].data
    data2 = fits.open(file2)[0].data

    wcs1 = WCS(fits.open(file1)[0].header)
    wcs2 = WCS(fits.open(file2)[0].header)

    return data1, data2, wcs1, wcs2


def get_freqs(wcs1, wcs2, freqlist1=None, freqlist2=None):

    print('')
    print('Getting frequencies...')
    print('')
    print(wcs1.array_shape)

    if freqlist1 == None:
        freqs1 = wcs1.all_pix2world(0,0,range(wcs1.array_shape[0]),0)[2]/1e6
    else:
        freqs1 = np.genfromtxt(freqlist1)/1e6

    if freqlist2 == None:
        freqs2 = wcs2.all_pix2world(0,0,range(wcs2.array_shape[0]),0)[2]/1e6
    else:
        freqs2 = np.genfromtxt(freqlist2)/1e6

    return freqs1, freqs2


def dragons_to_K(dragons, freq_dragons, out_dir):

    print('')
    print('Converting DRAGONS to K...')
    print('')

    KpJy = np.load('/srv/data/dva/conversion_factor_JytoK_July2024_BB.npy')

    f = interpolate.interp1d(KpJy[0], KpJy[1])
    KpJy_sub = np.expand_dims(f(freq_dragons),axis=(1,2))
    print(KpJy_sub.shape)

    dragons_K = dragons*KpJy_sub

    fig,ax = plt.subplots(2,2,figsize=(16,12))
    idx = 5
    ax[0,0].plot(KpJy[0],KpJy[1],zorder=0)
    ax[0,0].scatter(freq_dragons,KpJy_sub[:,0,0],s=5,color='k',zorder=1)
    ax[0,0].set_xlim(350,1050)
    ax[0,0].set_ylabel('K per Jy')
    ax[0,1].axis('off')

    im = ax[1,0].imshow(dragons[idx,:,:],origin='lower',vmin=-25,vmax=25,cmap='RdBu_r')
    ax[1,0].set_title(str(freq_dragons[idx])+' MHz, units: Jy/beam')
    fig.colorbar(im, ax=ax[1,0], shrink=0.8, pad=0.1, aspect=30, location='bottom')

    im = ax[1,1].imshow(dragons_K[idx,:,:],origin='lower',vmin=-2,vmax=2,cmap='RdBu_r')
    ax[1,1].set_title(str(freq_dragons[idx])+' MHz, units: K')
    fig.colorbar(im, ax=ax[1,1], shrink=0.8, pad=0.1, aspect=30, location='bottom')

    plt.savefig(out_dir+'K_conversion.png',transparent=False)
  
    return dragons_K


def dragons_nans(dragons, wcs_dragons):

    print('')
    print('NaN bad data in DRAGONS...')
    print('')

    decs = wcs_dragons.all_pix2world(0,range(dragons.shape[1]),0,0)[1]

    wbad1 = np.where(decs < -22)
    wbad2 = np.where(np.isnan(decs))
    print(decs[wbad1])
    print(decs[wbad2])

    dragons[:,wbad1,:] = np.nan
    dragons[:,wbad2,:] = np.nan
    
    return dragons


def regrid(data1, data2, wcs1, wcs2, out_dir):

    wcs1_2D = wcs1.dropaxis(2)
    wcs2_2D = wcs2.dropaxis(2)

    print('')
    print('Regridding '+str(data2.shape[2])+' to '+str(data1.shape[2])+'...')
    print('')

    data2_regrid = []
    for i in tqdm(range(0,data2.shape[0])):
    #for i in range(0,1):
        #print(i)
        regrid_slice, footprint = reproject_interp((data2[i,:,:],wcs2_2D),wcs1_2D)
        data2_regrid.append(regrid_slice)
    data2_regrid = np.array(data2_regrid)

    fig,ax = plt.subplots(1,2,figsize=(16,5))
    idx = 0
    ax[0].imshow(data2[idx,:,:],origin='lower',vmin=-2,vmax=2,cmap='RdBu_r')
    ax[1].imshow(data2_regrid[idx,:,:],origin='lower',vmin=-2,vmax=2,cmap='RdBu_r')
    plt.savefig(out_dir+'regridding.png',transparent=False)

    return data2_regrid


def match_freqs(data1, data2, freqs1, freqs2, out_dir):

    #freqs_new = np.linspace(400.5,729.5,330)
    freqs_new = np.linspace(512.5,729.5,218)
    #freqs_new = np.linspace(512.5,513.5,2)
    naxis1 = data1.shape[2]
    naxis2 = data1.shape[1]

    data1_new = np.empty([len(freqs_new),naxis2,naxis1])
    data2_new = np.empty([len(freqs_new),naxis2,naxis1])

    for i in tqdm(range(0,len(freqs_new))):

        fidx2 = np.where(abs(freqs2-freqs_new[i])<0.5)[0]
        fidx1 = np.where(abs(freqs1-freqs_new[i])<0.5)[0]

        #print(freqs_new[i],len(fidx2),len(fidx1))
        if len(fidx2) > 0:
            data2_new[i,:,:] = np.nanmedian(data2[fidx2,:,:],axis=0)
            #print('dragons')
        else:
            data2_new[i,:,:] = np.nan

        if len(fidx1) > 0:
            data1_new[i,:,:] = np.nanmedian(data1[fidx1,:,:],axis=0)
            #print('chime')
        else:
            data1_new[i,:,:] = np.nan

    idx = np.where(np.isfinite(data1_new[:,int(naxis2*0.75),int(naxis1/2)]))[0][0]
    print(freqs_new[idx])
    fig,ax = plt.subplots(1,2,figsize=(16,5))
    ax[0].imshow(data1_new[idx,:,:],origin='lower',vmin=-2,vmax=2,cmap='RdBu_r')
    ax[1].imshow(data2_new[idx,:,:],origin='lower',vmin=-2,vmax=2,cmap='RdBu_r')
    plt.savefig(out_dir+'compare_channels.png',transparent=False)


    return data1_new, data2_new, freqs_new


def make_fits_file_eq(data, freqs, out_dir, out_name):

    print('')
    print('Making FITS file...')
    print('')

    hdr = fits.PrimaryHDU().header

    del hdr['EXTEND']
    hdr['NAXIS']  = 3
    hdr['NAXIS1'] = data.shape[2]
    hdr['NAXIS2'] = data.shape[1]
    hdr['NAXIS3'] = data.shape[0]
    hdr['CRVAL1'] = 180.0
    hdr['CRVAL2'] = 0.0
    hdr['CRVAL3'] = freqs[0]
    hdr['CDELT1'] = -360/data.shape[2]
    hdr['CDELT2'] = 180/data.shape[1]
    hdr['CDELT3'] = freqs[1]-freqs[0]
    hdr['CRPIX1'] = data.shape[2]/2
    hdr['CRPIX2'] = data.shape[1]/2
    hdr['CRPIX3'] = 1
    hdr['CUNIT1'] = 'deg'
    hdr['CUNIT2'] = 'deg'
    hdr['CUNIT3'] = 'MHz'
    hdr['CTYPE1'] = 'RA---CAR'
    hdr['CTYPE2'] = 'DEC--CAR'
    hdr['CTYPE3'] = 'FREQ'

    fits.writeto(out_dir+out_name, data.astype('float32'), hdr, overwrite=True)

    return 


def mask_bright_srcs(chime, dragons, wcs, rmax=1):

    print('')
    print('Masking bright sources...')
    print('')

    cas_coord = SkyCoord('23h23m24.0s', '+58d48m54.0s', frame='icrs')
    cyg_coord = SkyCoord('19h59m28.4s', '+40d44m02.1s', frame='icrs')
    tau_coord = SkyCoord('5h34m31.9s', '+22d00m52.2s', frame='icrs')

    src_coords = [cas_coord,cyg_coord,tau_coord]

    decs_all = wcs.all_pix2world(0,range(dragons.shape[1]),0,0)[1]
    RAs_all = wcs.all_pix2world(range(dragons.shape[2]),0,0,0)[0]

    rara,decdec = np.meshgrid(RAs_all,decs_all)
    print(decdec.shape)
    print(rara.shape)
    
    for coord in src_coords:
        print(coord.ra.deg,coord.dec.deg)
        rRA  = (rara - coord.ra.deg)*np.cos(decdec*np.pi/180)
        rdec = decdec - coord.dec.deg
        r2D = np.sqrt(rdec**2+rRA**2)

        for ifrq in range(0,chime.shape[0]):
            chime[ifrq][r2D < rmax] = np.nan
            dragons[ifrq][r2D < rmax] = np.nan

    return chime, dragons

def mask_bad_RA(chime, dragons, wcs):

    print('')
    print('Masking bad RA region...')
    print('')

    decs_all = wcs.all_pix2world(0,range(dragons.shape[1]),0,0)[1]
    RAs_all = wcs.all_pix2world(range(dragons.shape[2]),0,0,0)[0]

    #wbad = np.where((RAs_all*24/360 > 17) & (RAs_all*24/360<21))
    wbad = np.where(RAs_all*24/360 > 17)

    for ifrq in range(0,chime.shape[0]):
        chime[ifrq][:,wbad] = np.nan
        dragons[ifrq][:,wbad] = np.nan

    return chime, dragons

def mask_bright_regions(chime, dragons):

    print('')
    print('Masking bright regions...')
    print('')

    ## *** For now using first frequency channel for brightness thresholding
    wbright_pi = np.where( np.sqrt(chime[0]**2+chime[0]**2) > 3)
    #wbright_i = np.where( chime[0] > 30)

    for ifrq in range(0,chime.shape[0]):

        chime[ifrq][wbright_pi] = np.nan
        dragons[ifrq][wbright_pi] = np.nan
        #chime[ifrq][wbright_i] = np.nan
        #dragons[ifrq][wbright_i] = np.nan

    return chime, dragons

def band_pass_slice(image, dxy, freq, R1min=0.1, R1max=0.3, R2min=5.0, R2max=5.5):
        
    image[np.isnan(image)] = 0.0

    image_FFT = np.fft.fft2(image)

    image_FFT_shift = np.fft.fftshift(image_FFT)

    uv_freq_m = np.fft.fftshift(np.fft.fftfreq(image.shape[1]))/dxy
    uv_freq_n = np.fft.fftshift(np.fft.fftfreq(image.shape[0]))/dxy
    #uv_m = uv_freq_m*(3e8/(freq*1e6))*180/(np.pi**2)
    #uv_n = uv_freq_n*(3e8/(freq*1e6))*180/(np.pi**2)
    uv_m = 2*uv_freq_m*(3e8/(freq*1e6))*180/np.pi
    uv_n = 2*uv_freq_n*(3e8/(freq*1e6))*180/np.pi

    xuv, yuv = np.meshgrid(uv_m, uv_n)
    ruv = np.sqrt(xuv**2+yuv**2)
    
    # high pass:
    b = np.pi/(R1max-R1min)
    r0 = (R1max+R1min)/2.
    mask1 = 0.5*np.sin(b*(ruv - r0))+0.5
    mask1[ruv < R1min] = 0
    mask1[ruv > R2min] = 0
    mask1[np.where((ruv > R1max) & (ruv < R2min))] = 1

    # low pass:
    b = np.pi/(R2max-R2min)
    r0 = (R2max+R2min)/2.
    mask2 = -0.5*np.sin(b*(ruv - r0))+0.5
    mask2[ruv > R2max] = 0
    mask2[ruv < R2min] = 0
 
    mask = mask1 + mask2
    
    image_FFT_shift = image_FFT_shift*mask
    image_filt = np.fft.ifft2(np.fft.ifftshift(image_FFT_shift))
  
    return image_filt, mask, uv_m, uv_n

def gf(mu,fwhm,x):
    return np.exp(-4*np.log(2)*((x-mu)**2)/fwhm**2)

def SA_deconvolve(image, R, dxy, freq):
    
    image[np.isnan(image)] = 0.0

    # image to uv plane:
    image_FFT = np.fft.fft2(image)
    image_FFT_shift = np.fft.fftshift(image_FFT)

    # make the beam in uv space:
    uv_freq_m = np.fft.fftshift(np.fft.fftfreq(image.shape[1]))/dxy
    uv_freq_n = np.fft.fftshift(np.fft.fftfreq(image.shape[0]))/dxy
    #uv_m = uv_freq_m*(3e8/(freq*1e6))*180/(np.pi**2)
    #uv_n = uv_freq_n*(3e8/(freq*1e6))*180/(np.pi**2)
    uv_m = 2*uv_freq_m*(3e8/(freq*1e6))*180/np.pi
    uv_n = 2*uv_freq_n*(3e8/(freq*1e6))*180/np.pi
    xuv, yuv = np.meshgrid(uv_m, uv_n)
    ruv = np.sqrt(xuv**2+yuv**2)
    SA_beam = gf(0,2*R,ruv)
    SA_beam[ruv>20] = 1
    
    # apply the deconvolution (division in uv plane)
    image_FFT_shift_new = np.empty_like(image_FFT_shift)
    image_FFT_shift_new = image_FFT_shift/SA_beam

    # low pass:
    Rmax = 5.5
    Rmin = 5
    b = np.pi/(Rmax-Rmin)
    r0 = (Rmax+Rmin)/2.
    mask = -0.5*np.sin(b*(ruv - r0))+0.5
    mask[ruv > Rmax] = 0
    mask[ruv < Rmin] = 1
    #mask = []

    image_deconv = np.fft.ifft2(np.fft.ifftshift(image_FFT_shift_new*mask))
    #image_deconv = np.fft.ifft2(np.fft.ifftshift(image_FFT_shift_new))

    return image_FFT_shift, mask, SA_beam, xuv, yuv, image_deconv.real


def dec_chime_vs_dragons(chime, dragons, wcs, freq_list, stokes, out_dir, 
                         ddec=1, dec_min=14, makeplots=False):
    
    print('')
    print('Calculating slopes and intercepts...')
    print('')

    decs = wcs.all_pix2world(0,range(dragons.shape[1]),0,0)[1]
    RAs  = wcs.all_pix2world(range(dragons.shape[2]),0,0,0)[0]
    idx_min = np.where(decs >= dec_min)[0][0]
    print(idx_min,decs[idx_min])

    slopes = np.empty([len(freq_list),len(decs)])
    ints   = np.empty([len(freq_list),len(decs)])
    slopes[:,:] = np.nan
    ints[:,:]   = np.nan

    decs_plot = decs[idx_min::][::8]
    print('plot decs:',decs_plot)

    for i in tqdm(range(idx_min,len(decs))):

        idx = np.where((decs <= decs[i]+ddec/2) & (decs >= decs[i]-ddec/2))[0]
        
        for j in range(0,len(freq_list)):
            if np.nanmax(dragons[j]) > 0.0:
                try:
                    wgood = np.where( np.isfinite(chime[j][idx]) & np.isfinite(dragons[j][idx]) )  
                    x = chime[j][idx][wgood].flatten()
                    y = dragons[j][idx][wgood].flatten()
                    fit = linregress(x,y)
                    slopes[j,i] = round(fit.slope,5)
                    ints[j,i] = round(fit.intercept,5)
                    corr = round(fit.rvalue,2)

                    if ((makeplots==True) & (decs[i] in decs_plot)):
                        make_TT_plot(chime[j][idx],dragons[j][idx],
                                    RAs,slopes[j,i],ints[j,i],decs[i],
                                    stokes,out_dir,freq_list[j],corr)

                except:
                    slopes[j,i] = np.nan
                    ints[j,i] = np.nan
            else:
                slopes[j,i] = np.nan
                ints[j,i] = np.nan

    np.savez(out_dir+'chime_vs_dragons_stokes'+stokes+'.npz', decs=decs, slopes=slopes, ints=ints, freqs=freq_list)

    return decs, slopes, ints


def dec_chime_vs_dragons_hpx(chime, dragons, wcs, freq_list, stokes, out_dir, 
                             nside=64, dec_min=12, makeplots=False):
    
    print('')
    print('Calculating slopes and intercepts...')
    print('')

    #freqs_plot = freq_list[::8]
    freqs_plot = freq_list[0]
    print('plot frequencies:',freqs_plot)

    pix = np.arange(0,hp.pixelfunc.nside2npix(nside),1)
    ra_hpx, dec_hpx = hp.pixelfunc.pix2ang(64, pix, nest=False, lonlat=True)

    dec_slices = np.unique(dec_hpx)[np.unique(dec_hpx) >= dec_min]
    print(len(dec_slices))

    slopes = np.empty([len(freq_list),len(dec_slices)])
    ints   = np.empty([len(freq_list),len(dec_slices)])
    slopes[:,:] = np.nan
    ints[:,:]   = np.nan

    for j in tqdm(range(0,len(freq_list))):
        
        chime_hpx, footprint   = reproject.reproject_to_healpix((chime[j], wcs.dropaxis(2)),coord_system_out='C',nside=nside)
        dragons_hpx, footprint = reproject.reproject_to_healpix((dragons[j], wcs.dropaxis(2)),coord_system_out='C',nside=nside)

        if np.nanmax(dragons_hpx) > 0.0:
            
            print(freq_list[j])

            for i in range(0,len(dec_slices)):
                
                idx = np.where(dec_hpx == dec_slices[i])
                #print(i,dec_hpx[idx][0],len(dec_hpx[idx]))

                try:
                    wgood = np.where( np.isfinite(chime_hpx[idx]) & np.isfinite(dragons_hpx[idx]) )  
                    x = chime_hpx[idx][wgood]#.flatten()
                    y = dragons_hpx[idx][wgood]#.flatten()
                    fit = linregress(x,y)
                    slopes[j,i] = round(fit.slope,5)
                    ints[j,i] = round(fit.intercept,5)
                    corr = round(fit.rvalue,2)
                    
                    #if ((makeplots==True) & (freq_list[j] in np.array(freqs_plot))):
                    if makeplots:
                        #print('make plots')
                        make_TT_plot_hpx(chime_hpx[idx],dragons_hpx[idx],
                                        ra_hpx[idx],slopes[j,i],ints[j,i],dec_slices[i],
                                        stokes,out_dir,freq_list[j],corr)
                except:
                    slopes[j,i] = np.nan
                    ints[j,i] = np.nan

        else:
            slopes[j,:] = np.nan
            ints[j,:] = np.nan

    np.savez(out_dir+'chime_vs_dragons_stokes'+stokes+'.npz', decs=dec_slices, slopes=slopes, ints=ints, freqs=freq_list)

    return dec_slices, slopes, ints


def make_TT_plot(chime_plot,dragons_plot,RAs,slope_plot,int_plot,
                 dec_plot,stokes,out_dir,freq_plot,corr):

    fig,ax = plt.subplots(1,1,figsize=(8,8))

    x = np.arange(-10,11,1)
    y = int_plot + slope_plot*x

    for ii in range(0,chime_plot.shape[0]):
        im = ax.scatter(chime_plot[ii],dragons_plot[ii],s=5,c=RAs*24/360,cmap='viridis')
    
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.grid()
    ax.plot(x,y,color='k')
    ax.set_xlabel('CHIME (Jy/beam)')
    ax.set_ylabel('DRAGONS (K)')
    ax.set_title('Stokes '+str(stokes).upper()+', dec='+str(dec_plot)+', freq='+str(freq_plot)+' MHz')
    ax.set_aspect('equal')
    ax.text(-1.9,1.75,'R='+str(corr))
    ax.text(-1.9,1.6,'CHIME [K] ='+str(slope_plot)+r'$\times$CHIME [Jy/beam] +'+str(int_plot)+' [K]')

    #fig.colorbar(im, ax=ax[1,1], shrink=0.8, pad=0.1, aspect=30, location='bottom')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.1, aspect=25, location='right',
                        ticks=[0,3,6,9,12,15,18,21,24])
    cbar.set_label('RA [hours]')

    plt.savefig(out_dir+'TT_plot_Stokes'+str(stokes)+'dec'+str(dec_plot)+'freq'+str(freq_plot)+'.png',facecolor='white')
    plt.close()

    return


def make_TT_plot_hpx(chime_plot,dragons_plot,RAs,slope_plot,int_plot,
                 dec_plot,stokes,out_dir,freq_plot,corr):


    fig,ax = plt.subplots(1,1,figsize=(8,8))

    x = np.arange(-10,11,1)
    y = int_plot + slope_plot*x

    im = ax.scatter(chime_plot,dragons_plot,s=5,c=RAs*24/360,cmap='viridis')
    
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.grid()
    ax.plot(x,y,color='k')
    ax.set_xlabel('CHIME (Jy/beam)')
    ax.set_ylabel('DRAGONS (K)')
    ax.set_title('Stokes '+str(stokes).upper()+', dec='+str(dec_plot)+', freq='+str(freq_plot)+' MHz')
    ax.set_aspect('equal')
    ax.text(-1.9,1.75,'R='+str(corr))
    ax.text(-1.9,1.6,'CHIME [K] ='+str(slope_plot)+r'$\times$CHIME [Jy/beam] +'+str(int_plot)+' [K]')

    #fig.colorbar(im, ax=ax[1,1], shrink=0.8, pad=0.1, aspect=30, location='bottom')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.1, aspect=25, location='right',
                        ticks=[0,3,6,9,12,15,18,21,24])
    cbar.set_label('RA [hours]')

    plt.savefig(out_dir+'TT_plot_Stokes'+str(stokes)+'dec'+str(dec_plot)+'freq'+str(freq_plot)+'.png',facecolor='white')
    plt.close()

    return


def smooth_conversion(conversion_file, out_dir, stokes, decs):

    print('')
    print('Smoothing conversion factor to CHIME dec gridding...')
    print('')

    params = np.load(conversion_file)
    decs_hpx   = params['decs']
    slopes_hpx = params['slopes']
    ints_hpx   = params['ints']
    freq_list  = params['freqs']

    print(decs_hpx.shape)
    print(decs.shape)
    print(slopes_hpx.shape)
    print(ints_hpx.shape)

    slopes_smooth = np.empty([len(freq_list),len(decs)])
    ints_smooth   = np.empty([len(freq_list),len(decs)])

    idx_use = np.where((decs_hpx >= 15) & (decs_hpx <= 85))[0]
    #print(decs_hpx[idx_use])

    for i in range(0,len(freq_list)):

        coeffs_slope = np.polyfit(decs_hpx[idx_use], slopes_hpx[i][idx_use], 14)
        coeffs_int   = np.polyfit(decs_hpx[idx_use], ints_hpx[i][idx_use], 14)

        poly_slope = np.poly1d(coeffs_slope)
        poly_int   = np.poly1d(coeffs_int)

        slopes_smooth[i] = poly_slope(decs)
        ints_smooth[i]   = poly_int(decs)

        #f = interpolate.interp1d(decs_hpx, slopes_hpx[i],kind='cubic')
        #slopes_smooth[i][idx_interp] = f(decs[idx_interp])

        #f = interpolate.interp1d(decs_hpx, ints_hpx[i],kind='cubic')
        #ints_smooth[i][idx_interp] = f(decs[idx_interp])

    np.savez(out_dir+'chime_vs_dragons_stokes'+stokes+'_smooth.npz', decs=decs, slopes=slopes_smooth, ints=ints_smooth, freqs=freq_list)
    
    return slopes_smooth, ints_smooth

def feather(image, dxy, Rlim, feather_type, freq):
    
    print('Feathering for '+feather_type+' at '+str(freq)+'...')

    image[np.isnan(image)] = 0.0

    image_FFT = np.fft.fft2(image)
    image_FFT_shift = np.fft.fftshift(image_FFT)

    uv_freq_m = np.fft.fftshift(np.fft.fftfreq(image.shape[1]))/dxy
    uv_freq_n = np.fft.fftshift(np.fft.fftfreq(image.shape[0]))/dxy
 
    #uv_m = uv_freq_m*(3e8/610e6)*180/(np.pi**2)
    #uv_n = uv_freq_n*(3e8/610e6)*180/(np.pi**2)
    uv_m = 2*uv_freq_m*(3e8/(freq*1e6))*180/np.pi
    uv_n = 2*uv_freq_n*(3e8/(freq*1e6))*180/np.pi

    xuv, yuv = np.meshgrid(uv_m, uv_n)
    ruv = np.sqrt(xuv**2+yuv**2)
    
    b = np.pi/(Rlim[1]-Rlim[0])
    r0 = (Rlim[0]+Rlim[1])/2.

    if feather_type == 'AS':
        mask = 0.5*np.sin(b*(ruv - r0))+0.5
        mask[ruv > Rlim[1]] = 1
        mask[ruv < Rlim[0]] = 0
    if feather_type == 'SA':
        mask = -0.5*np.sin(b*(ruv - r0))+0.5
        mask[ruv > Rlim[1]] = 0
        mask[ruv < Rlim[0]] = 1

    
    image_FFT_shift = image_FFT_shift*mask
    image_feath = np.fft.ifft2(np.fft.ifftshift(image_FFT_shift))
  
    return image_feath, mask, uv_m, uv_n


if __name__ =='__main__': 
    run_all()