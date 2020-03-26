import numpy

from . import base


class LinearGlucoseErrorModelV1(base.BasePolynomialModelT):
    def __init__(self, *, independent_key:str='S', dependent_key:str='A365'):
        super().__init__(independent_key=independent_key, dependent_key=dependent_key, mu_degree=1, scale_degree=1)
        

class LogisticGlucoseErrorModelV1(base.BaseAsymmetricLogisticT):
    def __init__(self, *, independent_key:str='S', dependent_key:str='A365'):
        super().__init__(independent_key=independent_key, dependent_key=dependent_key, scale_degree=1)


class CDWBackscatterModelV1(base.BaseAsymmetricLogisticT):
    def __init__(self, *, independent_key:str='X', dependent_key:str='BS'):
        super().__init__(independent_key=independent_key, dependent_key=dependent_key, scale_degree=1)


class CDWAbsorbanceModelV1(base.BaseAsymmetricLogisticT):
    def __init__(self, *, independent_key:str='X', dependent_key:str='A600'):
        super().__init__(independent_key=independent_key, dependent_key=dependent_key, scale_degree=1)


class ODAbsorbanceModelV1(base.BaseAsymmetricLogisticT):
    def __init__(self, *, independent_key:str='OD600', dependent_key:str='A600'):
        super().__init__(independent_key=independent_key, dependent_key=dependent_key, scale_degree=1)
