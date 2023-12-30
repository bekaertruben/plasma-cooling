import numpy as np
from scipy.stats import rv_continuous
from scipy.special import erf, kn
from scipy.integrate import quad


import numpy as np
from scipy.stats import rv_continuous
from scipy.special import erf, kn
from scipy.integrate import quad

def _mj_pdf(x, T=0.3):
    """
    The PDF of the Maxwell-Juttner distribution.

    Args:
        x (float): the lorentz factor
        T (float): the temperature of the distribution in units of mc^2 / k_B

    Returns:
        float: The PDF of the Maxwell-Juttner distribution
    """
    if x <= 1:
        return 0

    N = 1 / (T * kn(2, 1/T))
    return N * x * np.sqrt(x**2 - 1) * np.exp(-x/ T)


def _mj_cdf(x, T):
    """
    The exact CDF of the Maxwell-Juttner distribution, calculated using numerical integration.

    Args:
        x (float): the lorentz factor
        T (float): the temperature of the distribution in units of mc^2 / k_B

    Returns:
        float: The CDF of the Maxwell-Juttner distribution
    """
    if x <= 1:
        return 0

    I = quad(_mj_pdf, 1, x, args=(T), epsabs=1e-5)[0]
    return I


def _mj_cdf_puiseux(x, T=0.3):
    """
    The Puiseux expansion of the CDF of the Maxwell-Juttner distribution
    Calculated to 8th order using Wolfram Alpha and integral-calculator.com

    This is only valid for low temperatures (T <~ 0.4),
    and also breaks down for large x, where the CDF should instead be approximated as 1.

    Args:
        x (float): the lorentz factor
        T (float): the temperature of the distribution in units of mc^2 / k_B

    Returns:
        float: The CDF of the Maxwell-Juttner distribution
    """

    N = 1 / (T * kn(2, 1/T))

    term1 = np.sqrt(np.pi * T) * erf(np.sqrt((x-1)/T)) * np.exp((x-1)/T) * (
            -103378275*T**8
            +37837800*T**7
            -17297280*T**6
            +10644480*T**5
            -10321920*T**4
            +27525120*T**3
            +62914560*T**2
            +33554432*T
        )

    term2 = np.sqrt(x-1) * (
            13056*T*x**7
            +(97920*T**2-127232*T)*x**6
            +(636480*T**3-820480*T**2+595712*T)*x**5
            +(3500640*T**4-4463680*T**3+3219328*T**2-1887488*T)*x**4
            +(15752880*T**5-19768320*T**4+14125696*T**3-8252928*T**2+5253376*T)*x**3
            +(55135080*T**6-67438800*T**5+47526336*T**4-27636864*T**3+17683840*T**2-23438080*T)*x**2
            +(137837700*T**7-160720560*T**6+110682000*T**5-63942912*T**4+41331520*T**3-56671488*T**2-47526656*T)*x
            +(206756550*T**8-213513300*T**7+140180040*T**6-80285040*T**5+53328096*T**4-79043392*T**3-81085312*T**2+8448*T)
        )

    return N * np.exp(-x/T) * (term1 + term2) / 2**(51/2)


class MaxwellJuttner(rv_continuous):
    """
    Describes the Maxwell-Jüttner distribution for the Lorentz factor
    See https://en.wikipedia.org/wiki/Maxwell-Jüttner_distribution
    In the low-temp limit, this is equal to the Maxwell-Boltzmann distribution

    Arguments
    ---------
    
    T: float
        the temperature of the distribution in units of mc^2 / k_B
    """

    def _pdf(self, x, T):
        return _mj_pdf(x, T)
    
    def _cdf(self, x, T):
        if x <= 1:
            return 0

        # use the Puiseux expansion for low temperatures
        if T < 0.33:
            if x > 5:
                return 1
            else:
                return _mj_cdf_puiseux(x, T)
        
        # else, use numerical integration
        return _mj_cdf(x, T)