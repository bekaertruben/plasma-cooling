import numpy as np
from mpmath import mp
from scipy.special import kn, binom, erf
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

# For some reason, mpmath is the only library with an incomplete gamma accepting negative inputs
@np.vectorize(otypes=[np.float64])
def uppergamma(a, x):
    """ Upper incomplete gamma function """
    return mp.gammainc(a, x, mp.inf)


def mj_cdf_puiseux(x, T=0.3):
    """
    The Puiseux expansion of the CDF of the Maxwell-Juttner distribution
    Calculated to 8th order using Wolfram Alpha and integral-calculator.com

    This is only valid for low temperatures (T <~ 0.3),
    and also breaks down for large x, where the CDF should instead be approximated as 1.

    Args:
        x (float): the lorentz factor
        T (float): the temperature of the distribution in units of mc^2 / k_B

    Returns:
        float: The CDF of the Maxwell-Juttner distribution
    """

    N = 1 / (T * kn(2, 1/T))

    T2 = T*T
    T3 = T2*T
    T4 = T3*T
    T5 = T4*T
    T6 = T5*T
    T7 = T6*T
    T8 = T7*T

    term1 = np.sqrt(np.pi * T) * erf(np.sqrt((x-1)/T)) * np.exp((x-1)/T) * (
            -103378275*T8
            +37837800*T7
            -17297280*T6
            +10644480*T5
            -10321920*T4
            +27525120*T3
            +62914560*T2
            +33554432*T
        )

    term2 = np.sqrt(x-1) * (
            13056*T*x**7
            +(97920*T2-127232*T)*x**6
            +(636480*T3-820480*T2+595712*T)*x**5
            +(3500640*T4-4463680*T3+3219328*T2-1887488*T)*x**4
            +(15752880*T5-19768320*T4+14125696*T3-8252928*T2+5253376*T)*x**3
            +(55135080*T6-67438800*T5+47526336*T4-27636864*T3+17683840*T2-23438080*T)*x**2
            +(137837700*T7-160720560*T6+110682000*T5-63942912*T4+41331520*T3-56671488*T2-47526656*T)*x
            +(206756550*T8-213513300*T7+140180040*T6-80285040*T5+53328096*T4-79043392*T3-81085312*T2+8448*T)
        )

    return N * np.exp(-x/T) * (term1 + term2) / 2**(51/2)


class MaxwellJuttnerDistribution:
    """
    Describes the Maxwell-Juttner distribution of relativistic particles (in Lorentz factor).
    The CDF is calculated through numerical integration of the PDF, or approximated.

    The approximation using a Laurent series in the integral is extremely accurate for T >> 1, even at order 1.
    At low temperatures, instead the Puiseux expansion of the CDF is used.

    Args:
        T (float): the temperature of the distribution in units of mc^2 / k_B
        approximation_order (int): the order of the approximation to use for the CDF
            If None, the CDF will be calculated numerically
    """

    def __init__(self, T:float, approximation_order:int=None):
        """ Calculates the normalisation factor and the coefficients for the approximation (if required) """
        self.T = T = float(T)
        self.approximation_order = approximation_order

        self.N = 1 / (self.T * kn(2, 1/self.T)) # Normalisation factor

        if approximation_order == None:
            return

        # Laurent series for x * sqrt(x^2 - 1)
        k = np.arange(int(approximation_order/2) + 2)
        laur_expons = -2 * (k - 1)
        laur_coeffs = (-1)**k * binom(1/2, k)
        
        T_coeffs = - T**(3 - 2*k)
        self.coeffs = laur_coeffs * T_coeffs
        self.ss = 3 - 2*k

    def pdf(self, gamma):
        """ The Probability Density Function of the Maxwell-Juttner distribution """
        gamma = np.asarray(gamma, dtype=float)
        result = np.zeros_like(gamma)
        ret0 = gamma <= 1
        
        _gamma = gamma[~ret0]
        result[~ret0] = _gamma * np.sqrt(_gamma**2 - 1) * np.exp(-_gamma / self.T)

        return self.N * result

    def cdf(self, gamma, method='gammas'):
        """ The Cumulative Distribution Function of the Maxwell-Juttner distribution
        Specify the method by which to calculate the CDF:
            - 'exact': numerically integrate the PDF
            - 'gammas': use the approximation in terms of incomplete gamma functions (accurate for T >> 1)
            - 'puiseux': use the Puiseux expansion of the CDF (accurate for T << 1)
        """
        assert method in ("exact", "gammas", "puiseux"), "method must be one of 'exact', 'gammas', 'puiseux'"
        gamma = np.asarray(gamma, dtype=float)
        result = np.zeros_like(gamma)
        ret0 = gamma <= 1

        # if no order is specified, use numerical integration
        if method=="exact" or self.approximation_order == None:
            result[~ret0] = [quad(self.pdf, 1, g)[0] for g in gamma[~ret0]]

        elif method=="puiseux":
            result[~ret0] = mj_cdf_puiseux(gamma[~ret0], T=self.T)
        
        elif method=="gammas":
            _gamma = gamma[~ret0]
            for c, s in zip(self.coeffs, self.ss):
                result[~ret0] += c * (uppergamma(s, _gamma / self.T) - uppergamma(s, 1 / self.T))
                # this could probably be optimised a bit to calculate only one gamma function
            result[~ret0] *= self.N

        return result

    def ppf(self, p, method='gammas'):
        """ The Percent Point Function (inverse CDF) of the Maxwell-Juttner distribution """
        p = np.asarray(p)
        assert len(p.shape) == 1, "p must be 1D array"

        # Use the expectation value of beta to get an initial guess for gamma
        # This is not the expectation value of gamma, but it should be near the steepest part of the CDF
        E_beta = 2 * self.T * (self.T + 1) * np.exp(-1 / self.T) / kn(2, 1 / self.T)
        gamma0 = 1 / np.sqrt(1 - E_beta**2)

        # This is approximately the standard deviation of gamma ( for T >> 1 )
        std_gamma = 0.58 * (gamma0 - 1)
        starting_points = (gamma0 - std_gamma/3, gamma0 + std_gamma/3)

        # Find the root of CDF(gamma) - p = 0
        results = np.zeros_like(p)
        for i, _p in enumerate(p):
            def loss(gamma):
                sqerr = (self.cdf(gamma, method=method) - _p)**2
                if gamma < 1:
                    sqerr += self.N * (1 - gamma) # we do not want gamma < 1
                return sqerr
            res = minimize_scalar(
                loss,
                # bounds=(1, np.inf),
                method='brent',
                bracket=starting_points
            )
            if not res.success:
                raise RuntimeError(f"Failed to find root of CDF(gamma) - p = 0 for p = {p}")
            results[i] = res.x
        
        return results

    def sample(self, size=1, method='gammas'):
        """ Sample from the Maxwell-Juttner distribution """
        return self.ppf(np.random.random(size=size), method=method)
