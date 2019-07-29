import collections
import numpy

try:
    import theano.tensor as tt
    HAS_THEANO = True
except ModuleNotFoundError:
    HAS_THEANO = False

try:
    import pymc3
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
    if not HAS_THEANO:
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


class ImportWarner:
    """Mock for an uninstalled package, raises `ImportError` when used."""
    __all__ = []

    def __init__(self, module_name):
        self.module_name = module_name

    def __getattr__(self, attr):
        raise ImportError(
            f'{self.module_name} is not installed. In order to use this function try:\npip install {self.module_name}'
        )


class DilutionPlan(dict):
    """Represents the result of a dilution series planning."""
    def __init__(self, *, xmin:float, xmax:float, R:int, C:int, stock:float, mode:str, vmax:float, min_transfer:float):
        """Plans a regularly-spaced dilution series with in very few steps.
    
        Args:
            xmin (float): lowest concentration value in the result
            xmax (float): highest concentration in the result
            R (int): number of rows in the MTP
            C (int): number of colums in the MTP
            stock (float): stock concentration (must be >= xmax)
            mode (str): either 'log' or 'linear'
            vmax (float): maximum possible volume in the MTP
            min_transfer (float): minimum allowed volume for transfer steps
        """
        # process arguments
        if stock < xmax:
            raise ValueError(f'Stock concentration ({stock}) must be >= xmax ({xmax})')
        N = R * C
    
        # determine target concentrations
        if mode == 'log':
            ideal_targets = numpy.exp(numpy.linspace(numpy.log(xmax), numpy.log(xmin), N))
        elif mode == 'linear':
            ideal_targets = numpy.linspace(xmax, xmin, N)
        else:
            raise ValueError('mode must be either "log" or "linear".')
        
        ideal_targets = ideal_targets.reshape((R, C), order='F')
    
        # collect preparation instructions for each columns
        # (column, dilution steps, prepared from, transfer volumes)
        instructions = []
        actual_targets = []
    
        # transfer from stock until the volume is too low
        for c in range(C):
            vtransfer = numpy.round(vmax * ideal_targets[:,c] / stock, 0)
            if all(vtransfer >= min_transfer):
                instructions.append(
                    (c, 0, 'stock', vtransfer)
                )
                # compute the actually achieved target concentration
                actual_targets.append(vtransfer / vmax * stock)
            else:
                break
        
        # prepare remaining columns by diluting existing ones
        for c in range(len(instructions), C):
            # find the first source column that can be used (with sufficient transfer volume)
            for src_c in range(0, len(instructions)):
                _, src_df, _, _ = instructions[src_c]
                vtransfer = numpy.ceil(ideal_targets[:,c] / actual_targets[src_c] * vmax)
                # take the leftmost column (least dilution steps) where the minimal transfer volume is exceeded
                if all(vtransfer >= min_transfer):
                    instructions.append(
                        # increment the dilution step counter
                        (c, src_df+1, src_c, vtransfer)
                    )
                    # compute the actually achieved target concentration
                    actual_targets.append(vtransfer / vmax * actual_targets[src_c])
                    break
            
        if len(actual_targets) < C:
            message = f'Impossible with current settings.' \
                f' Only {len(instructions)}/{C} colums can be prepared.'
            if mode == 'linear':
                message += ' Try switching to "log" mode.'
            raise ValueError(message)
    
        self.R = R
        self.C = C
        self.N = R * C
        self.ideal_x = ideal_targets
        self.x = numpy.array(actual_targets).T
        self.xmin = numpy.min(actual_targets)
        self.xmax = numpy.max(actual_targets)
        self.instructions = instructions
        self.vmax = vmax
        self.v_stock = numpy.sum([
            v
            for _, dsteps, src, v in instructions
            if dsteps == 0
        ])
        self.max_steps = max([
            dsteps
            for _, dsteps, _, _ in instructions
        ])

    def __repr__(self):
        output = f'Serial dilution plan ({self.xmin:.5f} to {self.xmax:.2f})' \
            f' from at least {self.v_stock} µl stock:'
        for c, dsteps, src, vtransfer in self.instructions:
            output += f'\r\n\tPrepare column {c} with {vtransfer} µl from '
            if dsteps == 0:
                output += 'stock'
            else:
                output += f'column {src} ({dsteps} serial dilutions)'
        output += f'\r\n\tFill up all colums to {self.vmax} µl before aspirating.'
        return output
