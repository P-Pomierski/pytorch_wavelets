import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from pytorch_wavelets.utils import reflect
import pywt

def roll(x, n, dim, make_even=False):
    if n < 0:
        n = x.shape[dim] + n

    if make_even and x.shape[dim] % 2 == 1:
        end = 1
    else:
        end = 0

    if dim == 0:
        return torch.cat((x[-n:], x[:-n+end]), dim=0)
    elif dim == 1:
        return torch.cat((x[:,-n:], x[:,:-n+end]), dim=1)
    elif dim == 2 or dim == -3:
        return torch.cat((x[:,:,-n:], x[:,:,:-n+end]), dim=2)
    elif dim == 3 or dim == -2:
        return torch.cat((x[:,:,:,-n:], x[:,:,:,:-n+end]), dim=3)
    elif dim == 4 or dim == -1:
        return torch.cat((x[:,:,:,:,-n:], x[:,:,:,:,:-n+end]), dim=4)


def mypad(x, pad, mode='constant', value=0):
    """ Function to do numpy like padding on tensors. Only works for 2-D
    padding.

    Inputs:
        x (tensor): tensor to pad
        pad (tuple): tuple of (left, right, top, bottom, front, back) pad sizes
        mode (str): 'symmetric', 'wrap', 'constant, 'reflect', 'replicate', or
            'zero'. The padding technique.
    """
    if mode == 'symmetric':
        # Vertical only
        if pad[0] == 0 and pad[1] == 0 and pad[4] == 0 and pad[5] == 0:
            m1, m2 = pad[2], pad[3]
            l = x.shape[-2]
            xe = reflect(np.arange(-m1, l+m2, dtype='int32'), -0.5, l-0.5)
            return x[:,:,:,xe]
        # Horizontal only
        elif pad[2] == 0 and pad[3] == 0 and pad[4] == 0 and pad[5] == 0:
            m1, m2 = pad[0], pad[1]
            l = x.shape[-3]
            xe = reflect(np.arange(-m1, l+m2, dtype='int32'), -0.5, l-0.5)
            return x[:,:,xe]
        # Depth only
        elif pad[0] == 0 and pad[1] == 0 and pad[2] == 0 and pad[3] == 0:
            m1, m2 = pad[4], pad[5]
            l = x.shape[-1]
            xe = reflect(np.arange(-m1, l+m2, dtype='int32'), -0.5, l-0.5)
            return x[:,:,:,:,xe]
    elif mode == 'periodic':
        # Vertical only
        if pad[0] == 0 and pad[1] == 0 and pad[4] == 0 and pad[5] == 0:
            xe = np.arange(x.shape[-2])
            xe = np.pad(xe, (pad[2], pad[3]), mode='wrap')
            return x[:,:,:,xe]
        # Horizontal only
        elif pad[2] == 0 and pad[3] == 0 and pad[4] == 0 and pad[5] == 0:
            xe = np.arange(x.shape[-3])
            xe = np.pad(xe, (pad[0], pad[1]), mode='wrap')
            return x[:,:,xe]
        # Depth only
        elif pad[0] == 0 and pad[1] == 0 and pad[2] == 0 and pad[3] == 0:
            xe = np.arange(x.shape[-1])
            xe = np.pad(xe, (pad[4], pad[5]), mode='wrap')
            return x[:,:,:,:,xe]

    elif mode == 'constant' or mode == 'reflect' or mode == 'replicate':
        pad = (pad[4], pad[5], pad[2], pad[3], pad[0], pad[1])
        return F.pad(x, pad, mode, value)
    elif mode == 'zero':
        pad = (pad[4], pad[5], pad[2], pad[3], pad[0], pad[1])
        return F.pad(x, pad)
    else:
        raise ValueError("Unkown pad type: {}".format(mode))


def afb1d3d(x, h0, h1, mode='zero', dim=-1):
    """ 1D analysis filter bank (along one dimension only) of an image

    Inputs:
        x (tensor): 5D input with the last three dimensions the spatial input
        h0 (tensor): 5D input for the lowpass filter. Should have shape (1, 1,
            h, 1, 1), (1, 1, 1, w , 1) or (1, 1, 1, 1, d)
        h1 (tensor): 5D input for the highpass filter. Should have shape (1, 1,
            h, 1, 1), (1, 1, 1, w , 1) or (1, 1, 1, 1, d)
        mode (str): padding method
        dim (int) - dimension of filtering. d=2 is for a vertical filter (called
            column filtering but filters across the rows). d=3 is for a
            horizontal filter, (called row filtering but filters across the
            columns). d=4 is for the depth filtering but filters 

    Returns:
        lohi: lowpass and highpass subbands concatenated along the channel
            dimension
    """
    C = x.shape[1]
    # Convert the dim to positive
    d = dim % 5

    if d == 3:
        s = (1, 2, 1)
    elif d == 2:
        s = (2, 1, 1)
    else:
        s = (1, 1, 2)

    N = x.shape[d]
    # If h0, h1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(h0, torch.Tensor):
        h0 = torch.tensor(np.copy(np.array(h0).ravel()[::-1]),
                          dtype=torch.float, device=x.device)
    if not isinstance(h1, torch.Tensor):
        h1 = torch.tensor(np.copy(np.array(h1).ravel()[::-1]),
                          dtype=torch.float, device=x.device)
    L = h0.numel()
    L2 = L // 2
    shape = [1,1,1,1,1]
    shape[d] = L
    # If h aren't in the right shape, make them so
    if h0.shape != tuple(shape):
        h0 = h0.reshape(*shape)
    if h1.shape != tuple(shape):
        h1 = h1.reshape(*shape)
    h = torch.cat([h0, h1] * C, dim=0)

    if mode == 'per' or mode == 'periodization':
        if x.shape[dim] % 2 == 1:
            if d == 2:
                x = torch.cat((x, x[:,:,-1:]), dim=2)
            elif d ==3:
                x = torch.cat((x, x[:,:,:,-1:]), dim=3)
            else:
                x = torch.cat((x, x[:,:,:,:,-1:]), dim=4)
            N += 1
        x = roll(x, -L2, dim=d)

        if d == 3:
            pad = (0, L-1, 0)
        elif d == 2:
            pad = (L-1, 0, 0)
        else:
            pad = (0, 0, L-1)
        
        lohi = F.conv3d(x, h, padding=pad, stride=s, groups=C)
        N2 = N//2
        if d == 2:
            lohi[:,:,:L2] = lohi[:,:,:L2] + lohi[:,:,N2:N2+L2]
            lohi = lohi[:,:,:N2]
        elif d == 3:
            lohi[:,:,:,:L2] = lohi[:,:,:,:L2] + lohi[:,:,:,N2:N2+L2]
            lohi = lohi[:,:,:,:N2]
        else:
            lohi[:,:,:,:,:L2] = lohi[:,:,:,:,:L2] + lohi[:,:,:,:,N2:N2+L2]
            lohi = lohi[:,:,:,:,:N2]
    else:
        # Calculate the pad size
        outsize = pywt.dwt_coeff_len(N, L, mode=mode)
        p = 2 * (outsize - 1) - N + L
        if mode == 'zero':
            # Sadly, pytorch only allows for same padding before and after, if
            # we need to do more padding after for odd length signals, have to
            # prepad
            if p % 2 == 1:
                if d == 3:
                    pad = (0, 0, 0, 1, 0, 0)
                elif d == 2:
                    pad = (0, 0, 0, 0, 0, 1)
                else:
                    pad = (0, 1, 0, 0, 0, 0)
                x = F.pad(x, pad)
            
            if d == 3:
                pad = (0, p//2, 0)
            elif d == 2:
                pad = (p//2, 0, 0)
            else:
                pad = (0, 0, p//2)

            # Calculate the high and lowpass
            lohi = F.conv3d(x, h, padding=pad, stride=s, groups=C)
        elif mode == 'symmetric' or mode == 'reflect' or mode == 'periodic':
            if d == 3:
                pad = (0, 0, p//2, (p+1)//2, 0, 0)
            elif d == 2:
                pad = (p//2, (p+1)//2, 0, 0, 0, 0)
            else:
                pad = (0, 0, 0, 0, p//2, (p+1)//2)
            x = mypad(x, pad=pad, mode=mode)
            lohi = F.conv3d(x, h, stride=s, groups=C)
        else:
            raise ValueError("Unkown pad type: {}".format(mode))

    return lohi


def sfb1d3d(lo, hi, g0, g1, mode='zero', dim=-1):
    """ 1D synthesis filter bank of an image tensor
    """
    C = lo.shape[1]
    d = dim % 5
    # If g0, g1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(g0, torch.Tensor):
        g0 = torch.tensor(np.copy(np.array(g0).ravel()),
                          dtype=torch.float, device=lo.device)
    if not isinstance(g1, torch.Tensor):
        g1 = torch.tensor(np.copy(np.array(g1).ravel()),
                          dtype=torch.float, device=lo.device)
    L = g0.numel()
    shape = [1,1,1,1,1]
    shape[d] = L
    N = 2*(lo.shape[d])
    # If g aren't in the right shape, make them so
    if g0.shape != tuple(shape):
        g0 = g0.reshape(*shape)
    if g1.shape != tuple(shape):
        g1 = g1.reshape(*shape)

    if d == 3:
        s = (1, 2, 1)
    elif d == 2:
        s = (2, 1, 1)
    else:
        s = (1, 1, 2)

    g0 = torch.cat([g0]*C,dim=0)
    g1 = torch.cat([g1]*C,dim=0)
    if mode == 'per' or mode == 'periodization':
        y = F.conv_transpose3d(lo, g0, stride=s, groups=C) + \
            F.conv_transpose3d(hi, g1, stride=s, groups=C)
        if d == 2:
            y[:,:,:L-2] = y[:,:,:L-2] + y[:,:,N:N+L-2]
            y = y[:,:,:N]
        elif d == 3:
            y[:,:,:,:L-2] = y[:,:,:,:L-2] + y[:,:,:,N:N+L-2]
            y = y[:,:,:,:N]
        else:
            y[:,:,:,:,:L-2] = y[:,:,:,:,:L-2] + y[:,:,:,:,N:N+L-2]
            y = y[:,:,:,:,:N]
        y = roll(y, 1-L//2, dim=dim)
    else:
        if mode == 'zero' or mode == 'symmetric' or mode == 'reflect' or \
                mode == 'periodic':
            
            if d == 3:
                pad = (0, L-2, 0)
            elif d == 2:
                pad = (L-2, 0, 0)
            else:
                pad = (0, 0, L-2)
        
            y = F.conv_transpose3d(lo, g0, stride=s, padding=pad, groups=C) + \
                F.conv_transpose3d(hi, g1, stride=s, padding=pad, groups=C)
        else:
            raise ValueError("Unkown pad type: {}".format(mode))

    return y