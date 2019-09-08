def ams(s, b):
    import numpy as np
    return np.sqrt(2*((s+b+10)*np.log(1+s/(b+10))-s))

