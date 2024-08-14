import torch.nn as nn
import pywt
import lowlevel as lowlevel
import torch
import numpy as np

class DWT3DForward(nn.Module):
    """ Performs a 3d DWT Forward decomposition of an image

    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        """
    def __init__(self, J=1, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
            h0_dep, h1_dep = h0_col, h1_col
        else:
            if len(wave) == 2:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = h0_col, h1_col
                h0_dep, h1_dep = h0_col, h1_col
            elif len(wave) == 6:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = wave[2], wave[3]
                h0_dep, h1_dep = wave[4], wave[5]


        # Prepare the filters
        filts = lowlevel.prep_filt_afb3d(h0_col, h1_col, h0_row, h1_row, h0_dep, h1_dep)
        self.register_buffer('h0_col', filts[0])
        self.register_buffer('h1_col', filts[1])
        self.register_buffer('h0_row', filts[2])
        self.register_buffer('h1_row', filts[3])
        self.register_buffer('h0_dep', filts[4])
        self.register_buffer('h1_dep', filts[5])
        self.J = J
        self.mode = mode

    def forward(self, x):
        """ Forward pass of the DWT.

        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in}, D_{in})`

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of length J with the first entry
                being the finest scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}, D_{in}')` and yh has shape
                :math:`list(N, C_{in}, 7, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LLH, LHL, LHH, HLL, HLH, HHL and HHH coefficients.

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid. (Description part is NOT updated 3D yet)
        """
        yh = []
        yl = x
        mode = lowlevel.mode_to_int(self.mode)

        # Do a multilevel transform
        for j in range(self.J):
            # Do 1 level of the transform
            yl, high = lowlevel.AFB3D.apply(
                yl, self.h0_row, self.h1_row, self.h0_col, self.h1_col, self.h0_dep, self.h1_dep, mode)
            yh.append(high)

        return yl, yh


class DWT3DInverse(nn.Module):
    """ Performs a 3d DWT Inverse reconstruction of an image

    Args:
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
    """
    def __init__(self, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0_col, g1_col = wave.rec_lo, wave.rec_hi
            g0_row, g1_row = g0_col, g1_col
            g0_dep, g1_dep = g0_col, g1_col
        else:
            if len(wave) == 2:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = g0_col, g1_col
                g0_dep, g1_dep = g0_col, g1_col
            elif len(wave) == 6:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = wave[2], wave[3]
                g0_dep, g1_dep = wave[4], wave[5]
        # Prepare the filters
        filts = lowlevel.prep_filt_sfb3d(g0_col, g1_col, g0_row, g1_row, g0_dep, g1_dep)
        self.register_buffer('g0_col', filts[0])
        self.register_buffer('g1_col', filts[1])
        self.register_buffer('g0_row', filts[2])
        self.register_buffer('g1_row', filts[3])
        self.register_buffer('g0_dep', filts[4])
        self.register_buffer('g1_dep', filts[5])
        self.mode = mode

    def forward(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
              yl is a lowpass tensor of shape :math:`(N, C_{in}, H_{in}',
              W_{in}, D_{in}')` and yh is a list of bandpass tensors of shape
              :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'', D_{in})`. I.e. should match
              the format returned by DWTForward

        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.

        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """
        yl, yh = coeffs
        ll = yl
        mode = lowlevel.mode_to_int(self.mode)

        # Do a multilevel inverse transform
        for h in yh[::-1]:
            if h is None:
                h = torch.zeros(ll.shape[0], ll.shape[1], 7, ll.shape[-3],
                                ll.shape[-2], ll.shape[-1], device=ll.device)

            # 'Unpad' added dimensions
            if ll.shape[-3] > h.shape[-3]:
                ll = ll[...,:-1,:,:]
            if ll.shape[-2] > h.shape[-2]:
                ll = ll[...,:-1,:]
            if ll.shape[-1] > h.shape[-1]:
                ll = ll[...,:-1]
            ll = lowlevel.SFB3D.apply(
                ll, h, self.g0_row, self.g1_row, self.g0_col, self.g1_col, self.g0_dep, self.g1_dep, mode)
        return ll


# x = torch.randn(1, 1, 256, 256, 256)
# xfd = DWT3DForward(J=2, wave='haar')
# l, h = xfd(x)
# packet = pywt.WaveletPacketND(data=x.squeeze(), wavelet='haar', mode='zero')
# idwt = DWT3DInverse(wave='haar')
# x_reconstructed = idwt((l, h))

# print(np.allclose(x.numpy(), x_reconstructed.numpy(), atol=1e-5))

# print(np.allclose(   l.squeeze().numpy(),    packet['aaa'].data, atol=1e-5))
# print(np.allclose(h[0].squeeze()[0].numpy(), packet['daa'].data, atol=1e-5))
# print(np.allclose(h[0].squeeze()[1].numpy(), packet['ada'].data, atol=1e-5))
# print(np.allclose(h[0].squeeze()[2].numpy(), packet['dda'].data, atol=1e-5))
# print(np.allclose(h[0].squeeze()[3].numpy(), packet['aad'].data, atol=1e-5))
# print(np.allclose(h[0].squeeze()[4].numpy(), packet['dad'].data, atol=1e-5))
# print(np.allclose(h[0].squeeze()[5].numpy(), packet['add'].data, atol=1e-5))
# print(np.allclose(h[0].squeeze()[6].numpy(), packet['ddd'].data, atol=1e-5))

