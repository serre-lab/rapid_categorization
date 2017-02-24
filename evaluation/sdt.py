from scipy.stats import norm
from scipy.integrate import quad
from scipy import Inf
import numpy, scipy

def dprime_mAFC(Pc, m):
    """
    Compute d' corresponding to a certain proportion of correct
    responses in m-AFC tasks.

    Parameters
    ----------
    Pc : float
        Proportion of correct responses.
    m : int
        Number of alternatives.

    Returns
    -------
    dprime : float
        d' value

    Examples
    --------
    >>> dp = dprime_mAFC(0.7, 3)

    """

    if Pc < 0 or Pc > 1:
        raise ValueError("Pc must be between 0 and 1")
    if isinstance(m, int) == False:
        raise TypeError("m must be an int")

    def est_dp(dp):

        def pr(x):
            return (norm.pdf(x-dp) * (norm.cdf(x)**(m-1)))

        return (Pc - quad(pr, -Inf, Inf)[0])
    try:
        dprime = scipy.optimize.brentq(est_dp, -10, 10)#scipy.optimize.newton(est_dp, 1)
    except:
        dprime = numpy.nan

    return dprime