'''This module implements classes for higher-order spectral analysis
e.g. bispectrum, bicoherence, etc.
'''


# Standard library imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab

# Related 3rd-party imports
from ensemble import Ensemble
from utilities import get_timebase_indices
from nonparametric import _plot_image
import matplotlib.cbook as cbook

def apply_window(x, window, axis=0, return_window=None):
    '''
    Apply the given window to the given 1D or 2D array along the given axis.
    Parameters
    ----------
    x : 1D or 2D array or sequence
        Array or sequence containing the data.
    window : function or array.
        Either a function to generate a window or an array with length
        *x*.shape[*axis*]
    axis : integer
        The axis over which to do the repetition.
        Must be 0 or 1.  The default is 0
    return_window : bool
        If true, also return the 1D values of the window that was applied
    '''
    x = np.asarray(x)

    if x.ndim < 1 or x.ndim > 2:
        raise ValueError('only 1D or 2D arrays can be used')
    if axis+1 > x.ndim:
        raise ValueError('axis(=%s) out of bounds' % axis)

    xshape = list(x.shape)
    xshapetarg = xshape.pop(axis)

    if cbook.iterable(window):
        if len(window) != xshapetarg:
            raise ValueError('The len(window) must be the same as the shape '
                             'of x for the chosen axis')
        windowVals = window
    else:
        windowVals = window(np.ones(xshapetarg, dtype=x.dtype))

    if x.ndim == 1:
        if return_window:
            return windowVals * x, windowVals
        else:
            return windowVals * x

    xshapeother = xshape.pop()

    otheraxis = (axis+1) % 2

    windowValsRep = mlab.stride_repeat(windowVals, xshapeother, axis=otheraxis)

    if return_window:
        return windowValsRep * x, windowVals
    else:
        return windowValsRep * x

class Bispectrum(object):
    '''A class for bispectrum estimation.
    This class employs the methodology outlined in:
        Kim & Powers, "Digital bispectral analysis and
        its applications to nonlinear wave interactions",
        IEEE Trans. Plasma Sci. PS-7, 120 (1979);
    inquisitive readers are directed there for further details.
    For stationary signals `x` and `y`, the cross bispectrum `Bxy`
    is defined as
        Bxy(i, j) = E[ (X_{i + j})* Y_i Y_j]
    where X and Y are the FFTs of the kth realization of signals
    `x` and `y`, respectively, * denotes complex conjugation, and
    E[...] denotes the expectation value operator.
    For real-valued signals `x` and `y`, the following symmetry
    results from X_{-k} = (X_k)*:
        Bxy(i, j) = Bxy(j, i) = [Bxy(-i, -j)]*
    (i.e. see Eq. (15) from Kim & Powers; note that symmetry
    relation Eq. (16) only holds if x = y, however). Because
    only symmetry relation Eq. (15) holds, in general, the
    bispectrum needs to be computed over region A & B of
    Fig. 1(a) in Kim & Powers (i.e. the region defined by
    Eqs. (24a) and (24b) of Kim & Powers).
    Attributes:
    -----------
    Below, `x` and `y` refer to the signals for which the bispectrum
    is computed, and `Fs` is the signal sampling rate.
    Bxy - array_like, (`Nfr`, `Nfc`)
        The bispectrum estimate. Note that this corresponds to
        the "discrete bispectrum" rather than the bispectral
        density; the discrete bispectrum can be converted
        to the bispectral density via normalization by a
        a scalar factor related to the sampling rate and
        the number of points per realization.
        [Bxy] = [x] [y]^2
    b2xy - array_like, (`Nfr`, `Nfc`)
        The squared bicoherence estimate (defined by Eq. (19)
        of Kim & Powers). Note that 0 <= b2xy <= 1 and that
        larger values of b2xy indicate a larger degree of
        nonlinear (quadratic) coupling between the waves.
        [b2xy] = unitless
    Fs - float
        The signal sampling rate, as specified at initialization.
        [Fs] = arbitrary units
    frow - array_like, (`Nfr`,)
        The frequencies corresponding to the rows of `self.Bxy`
        and `self.b2xy`.
        [frow] = [Fs]
    fcol - array_like, (`Nfr`,)
        The frequencies corresponding to the columns of `self.Bxy`
        and `self.b2xy`.
        [frow] = [Fs]
    df - float
        The spacing of frequencies `self.frow` and `self.fcol`.
        [df] = [Fs]
    t - float
        The midpoint of the ensemble.
        [t] = 1 / [Fs]
    dt - float
        The temporal length of the ensemble.
        [dt] = 1 / [Fs]
    same_data - bool
        If True, the signals `x` and `y` provided at initialization
        were identical.
    Nreal - int
        The number of realizations used in the spectral estimates.
        The random error (i.e. variance) in both `Bxy` and `b2xy`
        decreases as ~ 1 / `Nreal` (Eqs. (28) & (31) of Kim & Powers).
    Npts_per_real - int
        The number of sample points per realization used in the
        computation of the spectral estimates.
    Npts_overlap - int
        The number of overlapping points between adjacent realizations
        in the computation of the spectral estimates.
    detrend - string
        The function applied to each realization before taking the FFT.
    window - callable or ndarray
        The window applied to each realization before taking the FFT.
    Methods:
    --------
    Type `help(Bispectrum)` in the IPython console for a listing.
    '''
    def __init__(self, x, y, Fs=1.0, t0=0.,
                 tlim=None, Npts_per_real=256,
                 detrend=None, window=mlab.window_hanning,
                 print_params=True, print_status=True):
        '''Create an instance of the `Bispectrum` class.
        Input Parameters:
        -----------------
        x, y - array_like, (`N`,)
            The signals for which the bispectrum `Bxy` and
            the squared bicoherence `b2xy` will be computed.
            A ValueError is raised if `x` or `y` are complex.
            A ValueError is raised if `x` and `y` contain a
            different number of samples. Obviously,`x` and `y`
            must be sampled at the *same* rate, `Fs`, to obtain
            sensible results.
            [x] = arbitrary units
            [y] = arbitrary units, potentially different than [x]
        Fs - float
            The sampling rate of `x` and `y`.
            If not specified, `Fs` is assigned a value of unity such that
            all frequencies are *normalized* to the sampling rate.
            [Fs] = arbitrary units
        t0 - float
            The initial time corresponding to `x[0]` (and `y[0]`).
            [t0] = 1 / [Fs]
        tlim - array_like, (2, M) or None
            If not None, then only use the portion of `x` and `y`
            that sit between `min(tlim[i])` and `max(tlim[i])`
            for 0 <= i < M. This is useful, for example,
            when spikes must be filtered from the signal.
            [tlim] = 1 / [Fs]
        Npts_per_real - int
            The number of sample points per realization. If None,
            `Tens` is used to compute `Npts_per_real` that is compatible
            with `Nreal_per_ens` and efficient FFT computation.
            If not None, `Tens` is ignored. A ValueError is raised
            if not a positive integer.
        detrend - string
            The function applied to each realization before taking FFT.
            May be [ 'default' | 'constant' | 'mean' | 'linear' | 'none']
            or callable, as specified in :py:func: `csd <matplotlib.mlab.csd>`.
            *Warning*: Naively detrending (even with something as simple as
            `mean` or `linear` detrending) can introduce detrimental artifacts
            into the computed spectrum, so *no* detrending is the default.
        window - callable or ndarray
            The window applied to each realization before taking FFT,
            as specified in :py:func: `csd <matplotlib.mlab.csd>`.
        print_params - bool
            If True, print relevant spectral parameters to screen.
        print_status - bool
            If True, print percentage of bispectrum and squared bicoherence
            computation complete.
        '''
        # Only real-valued signals are expected/supported at the moment
        if np.iscomplexobj(x) or np.iscomplexobj(y):
            raise ValueError('`x` and `y` must be real-valued signals!')

        if len(x) != len(y):
            raise ValueError('`x` and `y` must have the same length!')

        # Determine properties for ensemble averaging
        # (No use in overlap when `Nreal_per_ens` is unity).
        ens = Ensemble(
            x, Fs=Fs, t0=t0,
            Npts_per_real=Npts_per_real,
            Nreal_per_ens=1, Npts_overlap=0)

        # Record important aspects of computation
        self.same_data = x is y

        self.Fs = ens.Fs

        self.Npts_per_real = ens.Npts_per_real
        self.Npts_overlap = ens.Npts_overlap
        self.Npts_per_ens = ens.Npts_per_ens

        self.detrend = detrend
        self.window = window

        self.f = ens.f
        self.df = ens.df

        self.t = ens.t
        self.dt = ens.dt

        if print_params:
            self.printSpectralParams()

        # Compute frequencies for cross-bispectrum estimate noting that
        # the dimensions must span:
        #
        #   row:   -f_{Ny} <= f <= (0.5 * f_{Ny}), and
        #   column:      0 <= f <= f_{Ny},
        #
        # where f_{Ny} = (0.5 * `self.Fs`) is the Nyquist frequency.
        Nf = len(self.f)
        self.frow = np.concatenate((
            -self.f[::-1],
            self.f[1:((Nf // 2) + 1)]))
        self.fcol = self.f
        del self.f

        # Get FFTs
        Xk = ens.getFFTs(
            x,
            detrend=self.detrend,
            window=self.window)

        if not self.same_data:
            Yk = ens.getFFTs(
                y,
                detrend=self.detrend,
                window=self.window)
        else:
            Yk = Xk

        # Estimate bispectral quantities
        self._getBispectrum(Xk, Yk, tlim, print_status=print_status)
        self._getSquaredBicoherence(Xk, Yk, tlim, print_status=print_status)

    def printSpectralParams(self):
        print ('\ndt: %.6g' % self.dt)
        print ('df: %.6g' % self.df)
        print ('Npts_per_real: %i' % self.Npts_per_real)
        print ('overlap: %.2f'
               % (np.float(self.Npts_overlap) / self.Npts_per_real))
        print ('detrend: %s' % self.detrend)
        #print ('window: %s' % self.window.func_name)

        return

    def _getRealizationsBetween(self, tlim):
        'Get realizations between `tlim`, which is (2, M) array_like.'
        # Determine parameters of timebases corresponding to
        # (a) the start of a realization and (b) the end of a
        # realization
        Fs = 1. / self.dt
        Npts = len(self.t)
        t0a = self.t[0] - (0.5 * self.dt)
        t0b = self.t[0] + (0.5 * self.dt)

        ind = np.array([], dtype='int')

        # Loop through each temporal bound in `tlim`
        # (Maybe not the most efficient, but don't
        # expect this to be a bottleneck...)
        for i in np.arange(tlim.shape[1]):
            inda = get_timebase_indices(tlim[:, i], Fs, t0a, Npts)
            indb = get_timebase_indices(tlim[:, i], Fs, t0b, Npts)
            ind = np.concatenate((
                ind,
                np.intersect1d(inda, indb)))

        return ind

    def _getBispectrum(self, Xk, Yk, tlim, print_status=False):
        'Get bispectrum estimate.'
        # Get number of frequencies in computed one-sided FFTs.
        # When `self.Npts_per_real` is 2^N with integer N (as it is
        # constrained to be when simply specifying `Tens` at input),
        # then `Nf` is (2^{N - 1} + 1)
        Nf = len(self.fcol)

        # Initialize the bispectrum array
        shape = (len(self.frow), len(self.fcol))
        self.Bxy = np.zeros(shape, dtype='complex')

        # The row of `self.Bxy` corresponding to zero frequency
        # corresponds to array index `i0`
        i0 = Nf - 1

        if print_status:
            print ('')
            Np = (shape[0] * shape[1]) // 2
            p = 0.

        if tlim is not None:
            tind = self._getRealizationsBetween(tlim)
            self.Nreal = len(tind)
        else:
            tind = slice(None, None)
            self.Nreal = len(self.t)

        # Compute over region A from Fig. 1(a) of Kim & Powers
        for i in np.arange(0, (Nf // 2) + 1):  # i >= 0
            for j in np.arange(i, Nf - i):     # j >= 0
                if print_status:
                    print ('Bxy percent complete: %.1f \r' % (100 * (p / Np)))
                    p += 1.

                term1 = np.conj(Xk[i + j, ...])
                term2 = Yk[i, ...]
                term3 = Yk[j, ...]

                # Realization dimension (along axis -1) should have 1 element;
                # remove this dimension, as it causes problems with some
                # older versions of NumPy
                term1 = np.squeeze(term1, axis=-1)
                term2 = np.squeeze(term2, axis=-1)
                term3 = np.squeeze(term3, axis=-1)

                self.Bxy[i + i0, j] = np.mean(
                    term1[tind] * term2[tind] * term3[tind],
                    axis=-1)

        # Compute over region B from Fig. 1(a) of Kim & Powers
        for i in np.arange(-Nf + 1, 1):  # i <= 0
            for j in np.arange(-i, Nf):  # j >= 0 as i <= 0; |j| >= |i|
                # Need to account for boundary effects
                # to get accurate status, but let's just
                # enforce p <= Np and accept a few
                # fractions of a percent error in
                # the status message...
                if print_status and (p <= Np):
                    print ('Bxy percent complete: %.1f \r' % (100 * (p / Np)))
                    p += 1.

                # Note that Xk[i + j] = Xk[j - |i|] for i <= 0
                term1 = np.conj(Xk[j - np.abs(i), ...])

                # Note that Yk[i] = Yk[-|i|] = (Yk[|i|])* for i <= 0,
                # where z* indicates the complex conjugate of z
                term2 = np.conj(Yk[np.abs(i), ...])

                term3 = Yk[j, ...]

                # Realization dimension (along axis -1) should have 1 element;
                # remove this dimension, as it causes problems with some
                # older versions of NumPy
                term1 = np.squeeze(term1, axis=-1)
                term2 = np.squeeze(term2, axis=-1)
                term3 = np.squeeze(term3, axis=-1)

                self.Bxy[i + i0, j] = np.mean(
                    term1[tind] * term2[tind] * term3[tind],
                    axis=-1)

        return

    def _getSquaredBicoherence(self, Xk, Yk, tlim, print_status=False):
        'Get squared-bicoherence estimate.'
        # Get number of frequencies in computed one-sided FFTs.
        # When `self.Npts_per_real` is 2^N with integer N (as it is
        # constrained to be when simply specifying `Tens` at input),
        # then `Nf` is (2^{N - 1} + 1)
        Nf = len(self.fcol)

        # Initialize the bicoherence array
        shape = (len(self.frow), len(self.fcol))
        self.b2xy = np.zeros(shape)

        # The row of `self.Bxy` corresponding to zero frequency
        # corresponds to array index `i0`
        i0 = Nf - 1

        if print_status:
            print ('')
            Np = (shape[0] * shape[1]) // 2
            p = 0.

        if tlim is not None:
            tind = self._getRealizationsBetween(tlim)
        else:
            tind = slice(None, None)

        # Compute over region A from Fig. 1(a) of Kim & Powers
        for i in np.arange(0, (Nf // 2) + 1):  # i >= 0
            for j in np.arange(i, Nf - i):     # j >= 0
                if print_status:
                    print ('b2xy percent complete: %.1f \r' % (100 * (p / Np)))
                    p += 1.

                num = (np.abs(self.Bxy[i + i0, j])) ** 2

                # Realization dimension (along axis -1) should have 1 element;
                # thus, we don't need to specify an axis for the averaging
                denX = np.mean(
                    (np.abs(Xk[i + j, tind, ...])) ** 2)
                denY = np.mean(
                    (np.abs(Yk[i, tind, ...] * Yk[j, tind, ...])) ** 2)

                self.b2xy[i + i0, j] = num / (denX * denY)

        # Compute over region B from Fig. 1(a) of Kim & Powers
        for i in np.arange(-Nf + 1, 1):  # i <= 0
            for j in np.arange(-i, Nf):  # j >= 0 as i <= 0; |j| >= |i|
                # Need to account for boundary effects
                # to get accurate status, but let's just
                # enforce p <= Np and accept a few
                # fractions of a percent error in
                # the status message...
                if print_status and (p <= Np):
                    print ('b2xy percent complete: %.1f \r' % (100 * (p / Np)))
                    p += 1.

                num = (np.abs(self.Bxy[i + i0, j])) ** 2

                # Realization dimension (along axis -1) should have 1 element;
                # thus, we don't need to specify an axis for the averaging
                #
                # Note that Xk[i + j] = Xk[j - |i|] for i <= 0
                denX = np.mean(
                    (np.abs(Xk[j - np.abs(i), tind, ...])) ** 2)

                # Note that Yk[i] = Yk[-|i|] = (Yk[|i|])* for i <= 0,
                # where z* indicates the complex conjugate of z
                denY = np.mean(
                    (np.abs(np.conj(Yk[np.abs(i), tind, ...]) * Yk[j, tind, ...])) ** 2)

                self.b2xy[i + i0, j] = num / (denX * denY)

        if print_status:
            print ('')

        return

    def plotBispectrum(self, fnorm=1, vlim=None, cmap='viridis',
                       interpolation='none', fontsize=12,
                       title=None,
                       xlabel=r'$\mathregular{f_1}$',
                       ylabel=r'$\mathregular{f_2}$',
                       cblabel='$\mathregular{|B_{xy}(f_1, f_2)|}$',
                       cbticks=None,
                       cborientation='vertical',
                       ax=None, fig=None):
        'Plot bispectrum on a logarithmic scale.'

        if (ax is None) and (fig is None):
            fig = plt.figure(figsize=(6, 6))
            ax = plt.gca()

        # Only consider regions with finite (i.e. non-zero) bispectrum;
        # those with zero value are outside of the computation domain.
        Bxy_mag = np.abs(self.Bxy)
        Bxy_mag = np.ma.masked_where(Bxy_mag <= 0, Bxy_mag, copy=False)

        ax = _plot_image(
            self.fcol / fnorm, self.frow / fnorm, Bxy_mag,
            xlim=None, ylim=None, vlim=vlim,
            norm='log', cmap=cmap, interpolation=interpolation,
            title=title, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize,
            cblabel=cblabel, cbticks=None, cborientation=cborientation,
            ax=ax, fig=fig)

        ax.set_aspect('equal')
        plt.tight_layout()

        return ax

    def plotSquaredBicoherence(self, fnorm=1, vlim=None, cmap='viridis',
                               interpolation='none', fontsize=12,
                               title=None,
                               xlabel=r'$\mathregular{f_1}$',
                               ylabel=r'$\mathregular{f_2}$',
                               cblabel='$\mathregular{b^2_{xy}(f_1, f_2)}$',
                               cbticks=None,
                               cborientation='vertical',
                               ax=None, fig=None):
        'Plot bispectrum on a logarithmic scale.'

        if (ax is None) and (fig is None):
            fig = plt.figure(figsize=(6, 6))
            ax = plt.gca()

        # Only consider regions with finite (i.e. non-zero) bicoherence;
        # those with zero value are outside of the computation domain.
        b2xy = np.ma.masked_where(self.b2xy <= 0, self.b2xy)

        ax = _plot_image(
            self.fcol / fnorm, self.frow / fnorm, b2xy,
            xlim=None, ylim=None, vlim=vlim,
            norm=None, cmap=cmap, interpolation=interpolation,
            title=title, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize,
            cblabel=cblabel, cbticks=None, cborientation=cborientation,
            ax=ax, fig=fig)

        ax.set_aspect('equal')
        plt.tight_layout()

        return ax