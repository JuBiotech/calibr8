import numpy


from . import base


class LinearGlucoseErrorModelV1(base.BasePolynomialModelT):
    def __init__(self, independent_key, dependent_key, *, scale_degree:int=0, theta_names=None):
        super().__init__(independent_key=independent_key, dependent_key=dependent_key, mu_degree=1, scale_degree=scale_degree, theta_names=theta_names)
        

class LogisticGlucoseErrorModelV1(base.BaseAsymmetricLogisticT):
    def __init__(self, independent_key:str, dependent_key:str, *, scale_degree:int=0, theta_names=None):
        super().__init__(independent_key=independent_key, dependent_key=dependent_key, scale_degree=scale_degree, theta_names=theta_names)


class CDWBackscatterModelV1(base.BaseAsymmetricLogisticT):
    def __init__(self, dependent_key:str, independent_key:str='X', *, theta_names=None):
        super().__init__(independent_key=independent_key, dependent_key=dependent_key, scale_degree=1, theta_names=theta_names)


class CDWAbsorbanceModelV1(base.BaseAsymmetricLogisticT):
    def __init__(self, independent_key:str='X', dependent_key:str='A600', *, theta_names=None):
        super().__init__(independent_key=independent_key, dependent_key=dependent_key, scale_degree=1, theta_names=theta_names)


class ODAbsorbanceModelV1(base.BaseAsymmetricLogisticT):
    def __init__(self, independent_key:str='OD600', dependent_key:str='A600', *, theta_names=None):
        super().__init__(independent_key=independent_key, dependent_key=dependent_key, scale_degree=1, theta_names=theta_names)
