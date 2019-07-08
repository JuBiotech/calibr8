import collections
import numpy

try:
    import theano.tensor as tt
    HAS_PYMC3 = True
except ModuleNotFoundError:
    HAS_PYMC3 = False


def istensor(input:object):
    """"Convenience function to test whether an input is a TensorVariable
        or if an input array or list contains TensorVariables.
    
    Args:
        input: object to be tested
    
    Return: 
        result(bool): Indicates if the object is or in any instance contains a TensorVariable.
    """
    if not HAS_PYMC3:
        return False
    elif isinstance(input, str):
        return False
    elif isinstance(input, tt.TensorVariable):
        return True
    elif isinstance(input, dict):
        for element in input.values():
            if istensor(element):
                return True  
    elif isinstance(input, collections.Iterable):
        if len(input)>1:
            for element in input:
                if istensor(element):
                    return True
    return False
