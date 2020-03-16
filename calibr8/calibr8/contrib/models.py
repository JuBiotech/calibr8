import numpy


from . import base


class LinearGlucoseErrorModelV1(base.BasePolynomialModelT):
    def __init__(self, independent_key:str=None, dependent_key:str=None, *, scale_degree:int=0):
        super().__init__(independent_key=independent_key, dependent_key=dependent_key, mu_degree=1, scale_degree=scale_degree)
        

class LogisticGlucoseErrorModelV1(base.BaseAsymmetricLogisticT):
    def __init__(self, independent_key:str=None, dependent_key:str=None, *, scale_degree:int=0):
        super().__init__(independent_key=independent_key, dependent_key=dependent_key, scale_degree=scale_degree)


class CDWBackscatterModelV1(base.BaseAsymmetricLogisticT):
    def __init__(self, dependent_key:str='BS', independent_key:str='X'):
        super().__init__(independent_key=independent_key, dependent_key=dependent_key, scale_degree=1)


class CDWAbsorbanceModelV1(base.BaseAsymmetricLogisticT):
    def __init__(self, independent_key:str='X', dependent_key:str='A600'):
        super().__init__(independent_key=independent_key, dependent_key=dependent_key, scale_degree=1)


class ODAbsorbanceModelV1(base.BaseAsymmetricLogisticT):
    def __init__(self, independent_key:str='OD600', dependent_key:str='A600'):
        super().__init__(independent_key=independent_key, dependent_key=dependent_key, scale_degree=1)
