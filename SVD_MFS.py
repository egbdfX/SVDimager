import casacore.tables as tables
import scipy.io
import numpy
from sklearn.decomposition import PCA
from mat4py import loadmat
import scipy.io
from astropy.io import fits

C = 299792458

for ind in range(1):
    vis = tables.table('PSR'+str(ind)+'.ms', readonly = False)
    num_rows = vis.nrows()
    
    vis_data = vis.getcol('DATA', 0, num_rows)
    
    baselines = vis.getcol('UVW', 0, num_rows)
    
    flag = vis.getcol('FLAG', 0, num_rows)
    
    spw = tables.table(vis.getkeyword('SPECTRAL_WINDOW'), readonly=True)
    num_spw = spw.nrows()
    
    frequencies = spw.getcol('CHAN_FREQ')
    
    if 'WEIGHT_SPECTRUM' in vis.colnames():
        weight_spectrum = vis.getcol('WEIGHT_SPECTRUM', 0, num_rows)
        print("WEIGHT_SPECTRUM shape:", weight_spectrum.shape)
    else:
        print("No WEIGHT_SPECTRUM column found.")
        weight_spectrum = numpy.ones_like(vis_data, dtype=float)
    
    vis.close()
    
    nu = numpy.ravel(frequencies)

    u = baselines[:, 0:1]
    v = baselines[:, 1:2]
    w = baselines[:, 2:3]

    u_nu = u * nu / C
    v_nu = v * nu / C
    w_nu = w * nu / C
    
    baselines_nu = numpy.column_stack([
        u_nu.ravel(order='F'),
        v_nu.ravel(order='F'),
        w_nu.ravel(order='F')
    ])
    
    num = len(baselines_nu)
    rank = 3
    gro = 1
    M = numpy.zeros((gro,3))
    V = numpy.zeros((gro*3,3))
    B = numpy.zeros((num,gro*3))
    for k in range(gro):
        D = numpy.zeros((num,3))
        for i in range(num):
            D[i] = baselines_nu[i+k*num]
        M1 = numpy.mean(D, axis=0)
        pca = PCA(rank)
        pca.fit(D)
        V1 = pca.components_
        B1 = pca.transform(D)
        
        M[k] = M1
        V[2*k] = V1[0]
        V[2*k+1] = V1[1]
        V[2*k+2] = V1[2]
        for i in range(num):
            B[i][2*k] = B1[i][0]
            B[i][2*k+1] = B1[i][1]
            B[i][2*k+2] = B1[i][2]
    
    w0 = weight_spectrum[:, :, 0] * (~flag[:, :, 0]).astype(float)
    w3 = weight_spectrum[:, :, 3] * (~flag[:, :, 3]).astype(float)

    w_sum = w0 + w3

    safe_w = numpy.where(w_sum > 0, w_sum, 1.0)

    Visreal_2d = numpy.where(
        w_sum > 0,
        numpy.real(vis_data[:, :, 0] * w0 + vis_data[:, :, 3] * w3) / safe_w,
        0.0
    )
    Visimag_2d = numpy.where(
        w_sum > 0,
        numpy.imag(vis_data[:, :, 0] * w0 + vis_data[:, :, 3] * w3) / safe_w,
        0.0
    )

    Visreal = Visreal_2d.ravel(order='F')
    Visimag = Visimag_2d.ravel(order='F')
    
    Bin = B.copy()
    Min = M.copy()
    Vin = numpy.zeros((3, V.shape[1]))
    Vin = V.copy()
    
    M1 = numpy.dot(-Min, Vin[0, :])
    M2 = numpy.dot(-Min, Vin[1, :])
    M3 = numpy.dot(-Min, Vin[2, :])

    Bin[:, 0] = Bin[:, 0] - M1
    Bin[:, 1] = Bin[:, 1] - M2
    Bin[:, 2] = Bin[:, 2] - M3

    fits.writeto('Bin' + str(ind) + '.fits', Bin, overwrite=True)
    fits.writeto('Min' + str(ind) + '.fits', Min, overwrite=True)
    fits.writeto('Vin' + str(ind) + '.fits', Vin, overwrite=True)
    fits.writeto('Visreal' + str(ind) + '.fits', Visreal, overwrite=True)
    fits.writeto('Visimag' + str(ind) + '.fits', Visimag, overwrite=True)
