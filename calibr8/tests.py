import collections
import datetime
import logging
import matplotlib
from matplotlib import pyplot
import numpy
import pathlib
import pytest
import scipy
import scipy.stats as stats

import calibr8
import calibr8.utils


try:
    try:
        import pymc3 as pm
        import theano as backend
        import theano.tensor as at
    except ModuleNotFoundError:
        import pymc as pm
        import aesara as backend
        import aesara.tensor as at
    config = backend.config
    HAS_PYMC = True
except ModuleNotFoundError:
    HAS_PYMC = False

try:
    import pygmo
    HAS_PYGMO = True
except ModuleNotFoundError:
    HAS_PYGMO = False


dir_testfiles = pathlib.Path(pathlib.Path(__file__).absolute().parent, 'testfiles')


class _TestModel(calibr8.CalibrationModel):
    def __init__(self, independent_key=None, dependent_key=None, theta_names=None):
        if theta_names is None:
            theta_names = tuple('a,b,c'.split(','))
        super().__init__(independent_key='I', dependent_key='D', theta_names=theta_names)


class _TestPolynomialModel(calibr8.BasePolynomialModelT):
    def __init__(self, independent_key=None, dependent_key=None, theta_names=None, *, scale_degree=0, mu_degree=1):
        super().__init__(independent_key='I', dependent_key='D', mu_degree=mu_degree, scale_degree=scale_degree)


class _TestLogisticModel(calibr8.BaseAsymmetricLogisticT):
    def __init__(self, independent_key=None, dependent_key=None, theta_names=None, *, scale_degree=0):
        super().__init__(independent_key='I', dependent_key='D', scale_degree=scale_degree)


class _TestLogIndependentLogisticModel(calibr8.BaseLogIndependentAsymmetricLogisticT):
    def __init__(self, independent_key=None, dependent_key=None, theta_names=None, *, scale_degree=0):
        super().__init__(independent_key='I', dependent_key='D', scale_degree=scale_degree)


class TestBasicCalibrationModel:
    def test_init(self):
        em = _TestModel('I', 'D', theta_names=tuple('c,d,e'.split(',')))
        assert em.independent_key == 'I'
        assert em.dependent_key == 'D'
        print(em.theta_names)
        assert em.theta_names == ('c', 'd', 'e')
        assert em.theta_bounds is None
        assert em.theta_guess is None
        assert em.theta_fitted is None
        assert em.theta_timestamp is None
        assert em.cal_independent is None
        assert em.cal_dependent is None
        pass
    
    def test_constructor_signature_check(self):
        class EM_OK(calibr8.CalibrationModel):
            def __init__(self, arg1=1, *, kwonly=2, kwonlydefault=4):
                super().__init__('I', 'D', theta_names=tuple('abc'))
        EM_OK()

        class EM_args(calibr8.CalibrationModel):
            def __init__(self, arg1):
                super().__init__('I', 'D', theta_names=tuple('abc'))
        with pytest.raises(TypeError):
            EM_args(arg1=3)

        class EM_kwargs(calibr8.CalibrationModel):
            def __init__(self, *, kwonly, kwonlydefault=4):
                super().__init__('I', 'D', theta_names=tuple('abc'))
        with pytest.raises(TypeError):
            EM_kwargs(kwonly=3)
        
        pass

    def test_exceptions(self):
        independent = 'X'
        dependent = 'BS'
        x = numpy.array([1,2,3])
        y = numpy.array([4,5,6])
        cmodel = _TestModel()
        with pytest.raises(NotImplementedError):
            _ = cmodel.predict_dependent(x)
        with pytest.raises(NotImplementedError):
            _ = cmodel.predict_independent(x)
        with pytest.raises(NotImplementedError):
            _ = cmodel.infer_independent(y, lower=0, upper=10, steps=10, ci_prob=None)
        with pytest.raises(NotImplementedError):
            _ = cmodel.loglikelihood(y=y, x=x, theta=[1,2,3])
        pass

    def test_save_and_load_version_check(self):
        em = _TestModel()
        em.theta_guess = (1,1,1)
        em.theta_fitted = (1,2,3)
        em.theta_bounds = (
            (None, None),
            (0, 5),
            (0, 10)
        )

        # save and load
        em.save('save_load_test.json')
        em_loaded = _TestModel.load('save_load_test.json')

        # test version checking
        vactual = tuple(map(int, calibr8.__version__.split('.')))
        # increment patch
        calibr8.core.__version__ = f'{vactual[0]}.{vactual[1]}.{vactual[2]+1}'
        _TestModel.load('save_load_test.json')
        # increment minor version
        calibr8.core.__version__ = f'{vactual[0]}.{vactual[1]+1}.{vactual[2]}'
        _TestModel.load('save_load_test.json')
        # change major version
        calibr8.core.__version__ = f'{vactual[0]-1}.{vactual[1]}.{vactual[2]}'
        with pytest.raises(calibr8.MajorMismatchException):
            _TestModel.load('save_load_test.json')
        calibr8.core.__version__ = '.'.join(map(str, vactual))
        
        # load with the wrong model
        class DifferentEM(calibr8.CalibrationModel):
            pass

        with pytest.raises(calibr8.CompatibilityException):
            DifferentEM.load('save_load_test.json')
        return

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_save_and_load_attributes(self, ndim):
        shape = tuple(3 + numpy.arange(ndim))
        assert len(shape) == ndim

        em = _TestModel()
        em.theta_guess = (1,1,1)
        em.theta_fitted = (1,2,3)
        theta_timestamp = em.theta_timestamp
        em.theta_bounds = (
            (None, None),
            (0, 5),
            (0, 10)
        )
        em.cal_independent = numpy.random.uniform(0, 10, size=shape)
        em.cal_dependent = numpy.random.normal(size=len(em.cal_independent))

        # save and load
        em.save('save_load_test.json')
        em_loaded = _TestModel.load('save_load_test.json')

        assert isinstance(em_loaded, _TestModel)
        assert em_loaded.independent_key == em.independent_key
        assert em_loaded.dependent_key == em.dependent_key
        assert em_loaded.theta_bounds == em.theta_bounds
        assert em_loaded.theta_guess == em.theta_guess
        assert em_loaded.theta_fitted == em.theta_fitted
        assert em_loaded.theta_timestamp is not None
        assert em_loaded.theta_timestamp == theta_timestamp
        numpy.testing.assert_array_equal(em_loaded.cal_independent, em.cal_independent)
        numpy.testing.assert_array_equal(em_loaded.cal_dependent, em.cal_dependent)
        pass


class TestModelFunctions:
    def test_logistic(self):
        x = numpy.array([1.,2.,4.])
        theta = [2,2,4,1]
        expected = 2*2-4+(2*(4-2))/(1+numpy.exp(-2*1/(4-2)*(x-2)))
        true = calibr8.logistic(x, theta)
        assert (numpy.array_equal(true, expected))
        return

    def test_inverse_logistic(self):
        x = numpy.array([1.,2.,4.])
        theta = [2,2,4,1]
        forward = calibr8.logistic(x, theta)
        reverse = calibr8.inverse_logistic(forward, theta)
        assert (numpy.allclose(x, reverse))
        return

    def test_asymmetric_logistic(self):
        L_L = -3
        L_U = 4
        I_x = 4.5
        S = 3.3
        c = -1
        theta = (L_L, L_U, I_x, S, c)
        
        # test that forward and backward match
        x_test = numpy.linspace(I_x - 0.1, I_x + 0.1, 5)
        y_test = calibr8.asymmetric_logistic(x_test, theta)
        x_test_reverse = calibr8.inverse_asymmetric_logistic(y_test, theta)
        numpy.testing.assert_array_almost_equal(x_test_reverse, x_test)
        
        # test I_y
        assert (
            calibr8.asymmetric_logistic(I_x, theta)
            ==
            L_L + (L_U - L_L) * (numpy.exp(c) + 1) ** (-numpy.exp(-c))
        )

        # test slope at inflection point
        ϵ = 0.0001
        numpy.testing.assert_almost_equal(
            (calibr8.asymmetric_logistic(I_x + ϵ, theta) - calibr8.asymmetric_logistic(I_x - ϵ, theta)) / (2*ϵ),
            S
        )
        return

    def test_inverse_asymmetric_logistic(self):
        x = numpy.array([1.,2.,4.])
        theta = [0,4,2,1,1]
        forward = calibr8.asymmetric_logistic(x, theta)
        reverse = calibr8.inverse_asymmetric_logistic(forward, theta)
        numpy.testing.assert_allclose(x, reverse)
        return 

    def test_xlog_asymmetric_logistic(self):
        L_L = -2
        L_U = 2
        log_I_x = numpy.log10(1)
        S = 5
        c = -2
        theta = (L_L, L_U, log_I_x, S, c)

        # test that forward and backward match
        x_test = 10**(numpy.linspace(log_I_x - 1, log_I_x + 1, 200))
        y_test = calibr8.xlog_asymmetric_logistic(x_test, theta)
        x_test_reverse = calibr8.inverse_xlog_asymmetric_logistic(y_test, theta)
        numpy.testing.assert_array_almost_equal(x_test_reverse, x_test)
        
        # test I_y
        assert (
            calibr8.xlog_asymmetric_logistic(10**log_I_x, theta)
            ==
            L_L + (L_U - L_L) * (numpy.exp(c) + 1) ** (-numpy.exp(-c))
        )

        # test slope at inflection point
        ϵ = 0.0001
        x_plus = 10**log_I_x + ϵ
        x_minus = 10**log_I_x - ϵ
        y_plus = calibr8.xlog_asymmetric_logistic(x_plus, theta)
        y_minus = calibr8.xlog_asymmetric_logistic(x_minus, theta)
        # for the xlog model, the slope parameter refers to the 
        dy_dlogx = (y_plus - y_minus) / (numpy.log10(x_plus) - numpy.log10(x_minus))
        numpy.testing.assert_almost_equal(dy_dlogx, S)
        return

    def test_inverse_xlog_asymmetric_logistic(self):
        x = numpy.array([1., 2., 4.])
        theta = [0, 4, 2, 1, 1]
        forward = calibr8.xlog_asymmetric_logistic(x, theta)
        reverse = calibr8.inverse_xlog_asymmetric_logistic(forward, theta)
        numpy.testing.assert_allclose(x, reverse)
        return 

    def test_log_log_logistic(self):
        x = numpy.array([1.,2.,4.])
        theta = [2,2,4,1]
        expected = numpy.exp(2*2-4+(2*(4-2))/(1+numpy.exp(-2*1/(4-2)*(numpy.log(x)-2))))
        true = calibr8.log_log_logistic(x, theta)
        assert (numpy.array_equal(true, expected))
        expected = numpy.exp(calibr8.logistic(numpy.log(x), theta))   
        assert (numpy.array_equal(true, expected))
        return

    def test_inverse_log_log_logistic(self):
        x = numpy.array([1.,2.,4.])
        theta = [2,2,4,1]
        forward = calibr8.log_log_logistic(x, theta)
        reverse = calibr8.inverse_log_log_logistic(forward, theta)
        assert (numpy.allclose(x, reverse))
        return

    def test_xlog_logistic(self):
        x = numpy.array([1.,2.,4.])
        theta = [2,2,4,1]
        expected = 2*2-4+(2*(4-2))/(1+numpy.exp(-2*1/(4-2)*(numpy.log(x)-2)))
        true = calibr8.xlog_logistic(x, theta)
        assert (numpy.array_equal(true, expected))
        expected = calibr8.logistic(numpy.log(x), theta)
        assert (numpy.array_equal(true, expected))        
        return
        
    def test_inverse_xlog_logistic(self):
        x = numpy.array([1.,2.,4.])
        theta = [2,2,4,1]
        forward = calibr8.xlog_logistic(x, theta)
        reverse = calibr8.inverse_xlog_logistic(forward, theta)
        assert (numpy.allclose(x, reverse))
        return

    def test_ylog_logistic(self):
        x = numpy.array([1.,2.,4.])
        theta = [2,2,4,1]
        expected = numpy.exp(2*2-4+(2*(4-2))/(1+numpy.exp(-2*1/(4-2)*(x-2))))
        true = calibr8.ylog_logistic(x, theta)
        assert (numpy.array_equal(true, expected))
        expected = numpy.exp(calibr8.logistic(x, theta))
        assert (numpy.array_equal(true, expected))
        return

    def test_inverse_ylog_logistic(self):
        x = numpy.array([1.,2.,4.])
        theta = [2,2,4,1]
        forward = calibr8.ylog_logistic(x, theta)
        reverse = calibr8.inverse_ylog_logistic(forward, theta)
        assert (numpy.allclose(x, reverse))
        return


@pytest.mark.skipif(not HAS_PYMC, reason='requires PyMC')
class TestSymbolicModelFunctions:
    def _check_numpy_backend_equivalence(self, function, theta):
        # make sure that test value computation is turned off (PyMC likes to turn it on)
        with config.change_flags(compute_test_value='off'):
            # create computation graph
            x = at.vector('x', dtype=config.floatX)
            y = function(x, theta)
            assert isinstance(y, at.TensorVariable)

            # compile Theano/Aesara function
            f = backend.function([x], [y])
        
            # check equivalence of numpy and Theano/Aesara backend computation
            x_test = [1, 2, 4]
            numpy.testing.assert_almost_equal(
                f(x_test)[0],
                function(x_test, theta)
            )
        return

    def test_logistic(self):
        self._check_numpy_backend_equivalence(
            calibr8.logistic,
            [2, 2, 4, 1]
        )
        return

    def test_asymmetric_logistic(self):
        self._check_numpy_backend_equivalence(
            calibr8.asymmetric_logistic,
            [0, 4, 2, 1, 1]
        )
        return

    def test_log_log_logistic(self):
        self._check_numpy_backend_equivalence(
            calibr8.log_log_logistic,
            [2, 2, 4, 1]
        )
        return
    
    def test_xlog_logistic(self):
        self._check_numpy_backend_equivalence(
            calibr8.xlog_logistic,
            [2, 2, 4, 1]
        )
        return

    def test_ylog_logistic(self):
        self._check_numpy_backend_equivalence(
            calibr8.ylog_logistic,
            [2, 2, 4, 1]
        )
        return


class TestUtils:
    def test_datetime_parsing(self):
        assert calibr8.utils.parse_datetime(None) is None
        assert (
            calibr8.utils.parse_datetime('2018-12-01T09:27:30Z')
            ==
            datetime.datetime(2018, 12, 1, 9, 27, 30, tzinfo=datetime.timezone.utc)
        )
        assert (
            calibr8.utils.parse_datetime('2018-12-01T09:27:30+0000')
            ==
            datetime.datetime(2018, 12, 1, 9, 27, 30, tzinfo=datetime.timezone.utc)
        )

    def test_datetime_formatting(self):
        assert calibr8.utils.format_datetime(None) is None
        assert (
            calibr8.utils.format_datetime(datetime.datetime(2018, 12, 1, 9, 27, 30, tzinfo=datetime.timezone.utc))
            ==
            '2018-12-01T09:27:30Z'
        )

    @pytest.mark.skipif(HAS_PYMC, reason='run only if PyMC is not installed')
    def test_istensor_without_pymc(self):
        test_dict = {
            'a': 1, 
            'b': [1,2,3], 
            'c': numpy.array([(1,2), (3,4)])
        }
        assert not (calibr8.istensor(test_dict))
        assert not (calibr8.istensor(1.2))
        assert not (calibr8.istensor(-5))
        assert not (calibr8.istensor([1,2,3]))
        assert not (calibr8.istensor('hello'))

    @pytest.mark.skipif(not HAS_PYMC, reason='requires PyMC')
    def test_istensor_with_pymc(self):
        test_dict = {
            'a': 1, 
            'b': [1,2,3], 
            'c': numpy.array([(1,2), (3,4)])
        }
        assert not (calibr8.istensor(test_dict))
        assert not (calibr8.istensor(1.2))
        assert not (calibr8.istensor(-5))
        assert not (calibr8.istensor([1,2,3]))
        assert not (calibr8.istensor('hello'))
            
        test_dict2 = {
            'a': 1, 
            'b': [1,2,3], 
            'c': numpy.array([(1, at.as_tensor_variable([1,2,3])), (3,4)])
        }
        assert (calibr8.istensor(test_dict2))
        assert (calibr8.istensor([1, at.as_tensor_variable([1,2]), 3]))
        assert (calibr8.istensor(numpy.array([1, at.as_tensor_variable([1,2]), 3])))

    def test_import_warner(self):
        dummy = calibr8.utils.ImportWarner('dummy')
        with pytest.raises(ImportError):
            print(dummy.__version__)
        return

    @pytest.mark.skipif(HAS_PYMC, reason='run only if PyMC is not installed')
    def test_has_modules(self):
        assert not calibr8.utils.HAS_TENSORS
        assert not calibr8.utils.HAS_PYMC3
        return

    @pytest.mark.skipif(not HAS_PYMC, reason='requires PyMC')
    def test_has_modules(self):
        assert calibr8.utils.HAS_TENSORS
        assert calibr8.utils.HAS_PYMC3
        return

    def test_assert_version_match(self):
        # fist shorter
        calibr8.utils.assert_version_match("1", "1.2.2.2")
        calibr8.utils.assert_version_match("1.1", "1.1.2.2")
        calibr8.utils.assert_version_match("1.1.1", "1.1.1.2")
        calibr8.utils.assert_version_match("1.1.1.1", "1.1.1.1")
        # second shorter
        calibr8.utils.assert_version_match("1.2.2.2", "1")
        calibr8.utils.assert_version_match("1.1.2.2", "1.1")
        calibr8.utils.assert_version_match("1.1.1.2", "1.1.1")
        calibr8.utils.assert_version_match("1.1.1.1", "1.1.1.1")

        with pytest.raises(calibr8.MajorMismatchException):
            calibr8.utils.assert_version_match("1.1.1.1", "2.1.1.1")
        with pytest.raises(calibr8.MinorMismatchException):
            calibr8.utils.assert_version_match("1.1.1.1", "1.2.1.1")
        with pytest.raises(calibr8.PatchMismatchException):
            calibr8.utils.assert_version_match("1.1.1.1", "1.1.2.1")
        with pytest.raises(calibr8.BuildMismatchException):
            calibr8.utils.assert_version_match("1.1.1.1", "1.1.1.2")
        return

    @pytest.mark.parametrize("residual_type", ["relative", "absolute"])
    def test_plot_model(self, residual_type):
        em = _TestPolynomialModel(independent_key='S', dependent_key='A365', mu_degree=1, scale_degree=1)
        em.theta_fitted = [0, 2, 0.1, 1]
        em.cal_independent = numpy.linspace(0.1, 10, 7)
        mu, scale, df = em.predict_dependent(em.cal_independent)
        em.cal_dependent = scipy.stats.t.rvs(loc=mu, scale=scale, df=df)

        fig, axs = calibr8.utils.plot_model(em, residual_type=residual_type)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert numpy.shape(axs) == (3,)
        pyplot.close()
        pass

    def test_plot_model_band_xlim(self):
        cm = _TestPolynomialModel(independent_key='S', dependent_key='A365', mu_degree=1, scale_degree=1)
        cm.theta_fitted = [0, 2, 0.1, 1]
        cm.cal_independent = numpy.linspace(0.1, 10, 7)
        mu, scale, df = cm.predict_dependent(cm.cal_independent)
        cm.cal_dependent = scipy.stats.t.rvs(loc=mu, scale=scale, df=df)

        # fetch the default limits for comparison
        fig, axs = calibr8.utils.plot_model(cm, band_xlim=(3, 5))
        orig_0 = axs[0].get_xlim()
        orig_1 = axs[1].get_xlim()
        orig_2 = axs[2].get_xlim()
        pyplot.close()

        # no zooming in with tighter bands - data is always shown
        fig, axs = calibr8.utils.plot_model(cm, band_xlim=(3, 5))
        assert axs[0].get_xlim() == orig_0
        assert axs[1].get_xlim() == orig_1
        assert axs[2].get_xlim() == orig_2
        pyplot.close()

        # wider band limits only affect the first two subplots
        fig, axs = calibr8.utils.plot_model(cm, band_xlim=(0, 5))
        assert axs[0].get_xlim()[0] < orig_0[0]
        numpy.testing.assert_approx_equal(axs[0].get_xlim()[1], orig_0[1], significant=4)
        assert axs[1].get_xlim()[0] < orig_1[0]
        # the upper xlim in the logarithmic plot changes because of axis scaling
        # residuals are unaffected
        assert axs[2].get_xlim() == orig_2
        pyplot.close()
        pass


class TestContribBase:
    def test_cant_instantiate_base_models(self):
        with pytest.raises(TypeError):
            calibr8.BaseModelT(independent_key='I', dependent_key='D')
        with pytest.raises(TypeError):
            calibr8.BaseAsymmetricLogisticT(independent_key='I', dependent_key='D')
        with pytest.raises(TypeError):
            calibr8.BasePolynomialModelT(independent_key='I', dependent_key='D', mu_degree=1, scale_degree=1)
        pass

    def test_base_polynomial_t(self):
        for mu_degree, scale_degree in [(1, 0), (1, 1), (1, 2), (2, 0)]:
            theta_mu = (2.2, 1.2, 0.2)[:mu_degree+1]
            theta_scale = (3.1, 0.4, 0.2)[:scale_degree+1]
            theta = theta_mu + theta_scale + (1,)

            em = _TestPolynomialModel(independent_key='I', dependent_key='D', mu_degree=mu_degree, scale_degree=scale_degree)
            assert len(em.theta_names) == mu_degree+1 + scale_degree+1 + 1
            assert len(em.theta_names) == len(theta)

            x = numpy.linspace(0, 10, 3)
            mu, scale, df = em.predict_dependent(x, theta=theta)


            expected = numpy.polyval(theta_mu[::-1], x)
            numpy.testing.assert_array_equal(mu, expected)
            
            expected = numpy.polyval(theta_scale[::-1], mu)
            numpy.testing.assert_array_equal(scale, expected)

            numpy.testing.assert_array_equal(df, 1)
        pass

    def test_base_polynomial_t_inverse(self):
        for mu_degree, scale_degree in [(1, 0), (1, 1), (1, 2), (2, 0)]:
            theta_mu = (2.2, 1.2, 0.2)[:mu_degree+1]
            theta_scale = (3.1, 0.4, 0.2)[:scale_degree+1]
            theta = theta_mu + theta_scale + (1,)

            em = _TestPolynomialModel(independent_key='I', dependent_key='D', mu_degree=mu_degree, scale_degree=scale_degree)
            em.theta_fitted = theta
            assert len(em.theta_names) == mu_degree+1 + scale_degree+1 + 1
            assert len(em.theta_names) == len(theta)

            x = numpy.linspace(0, 10, 7)
            mu, scale, df = em.predict_dependent(x, theta=theta)

            if mu_degree < 2:
                x_inverse = em.predict_independent(mu)
                numpy.testing.assert_array_almost_equal(x_inverse, x)
            else:
                with pytest.raises(NotImplementedError):
                    em.predict_independent(mu)
        pass

    def test_base_asymmetric_logistic_t(self):
        for scale_degree in [0, 1, 2]:
            theta_mu = (-0.5, 0.5, 1, 1, -1)
            theta_scale = (3.1, 0.4, 0.2)[:scale_degree+1]
            theta = theta_mu + theta_scale + (1,)

            em = _TestLogisticModel(independent_key='I', dependent_key='D', scale_degree=scale_degree)
            assert len(em.theta_names) == 5 + scale_degree+1 + 1
            assert len(em.theta_names) == len(theta)

            x = numpy.linspace(1, 5, 3)
            mu, scale, df = em.predict_dependent(x, theta=theta)

            expected = calibr8.asymmetric_logistic(x, theta_mu)
            numpy.testing.assert_array_equal(mu, expected)
            
            expected = numpy.polyval(theta_scale[::-1], mu)
            numpy.testing.assert_array_equal(scale, expected)

            numpy.testing.assert_array_equal(df, 1)
        pass

    def test_base_asymmetric_logistic_t_inverse(self):
        for scale_degree in [0, 1, 2]:
            theta_mu = (-0.5, 0.5, 1, 1, -1)
            theta_scale = (3.1, 0.4, 0.2)[:scale_degree+1]
            theta = theta_mu + theta_scale + (1,)

            em = _TestLogisticModel(independent_key='I', dependent_key='D', scale_degree=scale_degree)
            em.theta_fitted = theta
            
            x = numpy.linspace(1, 5, 7)
            mu, scale, df = em.predict_dependent(x, theta=theta)

            x_inverse = em.predict_independent(mu)
            numpy.testing.assert_array_almost_equal(x_inverse, x)
        pass

    def test_base_xlog_asymmetric_logistic_t(self):
        for scale_degree in [0, 1, 2]:
            theta_mu = (-0.5, 0.5, 1, 1, -1)
            theta_scale = (3.1, 0.4, 0.2)[:scale_degree+1]
            theta = theta_mu + theta_scale + (1,)

            em = _TestLogIndependentLogisticModel(independent_key='I', dependent_key='D', scale_degree=scale_degree)
            assert len(em.theta_names) == 5 + scale_degree+1 + 1
            assert len(em.theta_names) == len(theta)

            x = numpy.linspace(1, 5, 3)
            mu, scale, df = em.predict_dependent(x, theta=theta)

            expected = calibr8.xlog_asymmetric_logistic(x, theta_mu)
            numpy.testing.assert_array_equal(mu, expected)
            
            expected = numpy.polyval(theta_scale[::-1], mu)
            numpy.testing.assert_array_equal(scale, expected)

            numpy.testing.assert_array_equal(df, 1)
        pass

    def test_base_xlog_asymmetric_logistic_t_inverse(self):
        for scale_degree in [0, 1, 2]:
            theta_mu = (-0.5, 0.5, 1, 1, -1)
            theta_scale = (3.1, 0.4, 0.2)[:scale_degree+1]
            theta = theta_mu + theta_scale + (1,)

            em = _TestLogIndependentLogisticModel(independent_key='I', dependent_key='D', scale_degree=scale_degree)
            em.theta_fitted = theta
            
            x = numpy.linspace(1, 5, 7)
            mu, scale, df = em.predict_dependent(x, theta=theta)

            x_inverse = em.predict_independent(mu)
            numpy.testing.assert_array_almost_equal(x_inverse, x)
        pass


class TestBasePolynomialModelT:
    def test_infer_independent(self):
        em = _TestPolynomialModel(independent_key='S', dependent_key='A365', mu_degree=1, scale_degree=1)
        em.theta_fitted = [0, 2, 0.1, 1]
        pst = em.infer_independent(y=1, lower=0, upper=20, steps=876)

        assert len(pst.eti_x) == len(pst.eti_pdf)
        assert len(pst.hdi_x) == len(pst.hdi_pdf)
        assert tuple(pst.eti_x[[0, -1]]) == (0, 20)
        assert tuple(pst.hdi_x[[0, -1]]) == (0, 20)
        assert (numpy.isclose(scipy.integrate.cumtrapz(pst.eti_pdf, pst.eti_x)[-1], 1, atol=0.0001))
        assert (numpy.isclose(scipy.integrate.cumtrapz(pst.hdi_pdf, pst.hdi_x)[-1], 1, atol=0.0001))
        assert pst.eti_lower == pst.hdi_lower == 0
        assert pst.eti_upper == pst.hdi_upper == 20
        assert pst.eti_prob == 1
        assert pst.hdi_prob == 1

        # check trimming to [2.5,97.5] interval
        pst = em.infer_independent(y=[1, 2], lower=0, upper=20, steps=1775, ci_prob=0.95)

        assert len(pst.eti_x) == len(pst.eti_pdf)
        assert len(pst.hdi_x) == len(pst.hdi_pdf)
        assert numpy.isclose(pst.eti_prob, 0.95, atol=0.0001)
        assert numpy.isclose(pst.hdi_prob, 0.95, atol=0.0001)
        assert numpy.isclose(scipy.integrate.cumtrapz(pst.eti_pdf, pst.eti_x)[-1], 0.95, atol=0.0001)
        assert numpy.isclose(scipy.integrate.cumtrapz(pst.hdi_pdf, pst.hdi_x)[-1], 0.95, atol=0.0001)
        assert pst.eti_lower == pst.eti_x[0]
        assert pst.eti_upper == pst.eti_x[-1]
        assert pst.hdi_lower == pst.hdi_x[0]
        assert pst.hdi_upper == pst.hdi_x[-1]

        # check that error are raised by wrong input
        with pytest.raises(ValueError):
            _ = em.infer_independent(y=1, lower=0, upper=20, steps=1000, ci_prob=(-1))
        with pytest.raises(ValueError):
            _ = em.infer_independent(y=1, lower=0, upper=20, steps=1000, ci_prob=(97.5))
        pass

    @pytest.mark.skipif(not HAS_PYMC, reason='requires PyMC')
    def test_symbolic_loglikelihood_checks_and_warnings(self):
        cmodel = _TestPolynomialModel(independent_key='S', dependent_key='A', mu_degree=1, scale_degree=1)
        cmodel.theta_fitted = [0, 1, 0.1, 1]
       
        # create test data
        x_true = numpy.array([1,2,3,4,5])
        y_obs = cmodel.predict_dependent(x_true)[0]

        with pm.Model():
            x_hat = pm.Uniform("x_hat", shape=5)
            with pytest.raises(ValueError, match="`name` must be specified"):
                cmodel.loglikelihood(x=x_hat, y=y_obs)

            with pytest.warns(DeprecationWarning, match="Use `name` instead"):
                cmodel.loglikelihood(x=x_hat, y=y_obs, replicate_id='A01', dependent_key='A')
        pass

    @pytest.mark.skipif(not HAS_PYMC, reason='requires PyMC')
    def test_symbolic_loglikelihood(self):
        cmodel = _TestPolynomialModel(independent_key='S', dependent_key='A', mu_degree=1, scale_degree=1)
        cmodel.theta_fitted = [0, 1, 0.1, 1]
       
        # create test data
        x_true = numpy.array([1,2,3,4,5])
        y_obs = cmodel.predict_dependent(x_true)[0]

        x_hat = at.vector()
        x_hat.tag.test_value = x_true
        L = cmodel.loglikelihood(x=x_hat, y=y_obs, name='L_A01_A')
        assert isinstance(L, at.TensorVariable)
        assert L.ndim == 0

        # compare the two loglikelihood computation methods
        x_test = numpy.random.normal(x_true, scale=0.1)
        actual = L.eval({
            x_hat: x_test
        })
        expected = cmodel.loglikelihood(x=x_test, y=y_obs)
        assert numpy.ndim(expected) == 0
        assert numpy.ndim(actual) == 0
        numpy.testing.assert_almost_equal(actual, expected, 6)
        pass

    @pytest.mark.skipif(not HAS_PYMC, reason='requires PyMC')
    def test_symbolic_loglikelihood_in_modelcontext(self):
        cmodel = _TestPolynomialModel(independent_key='S', dependent_key='A', mu_degree=1, scale_degree=1)
        cmodel.theta_fitted = [0, 0.5, 0.1, 1]
       
        # create test data
        x_true = numpy.array([1,2,3,4,5])
        y_obs = cmodel.predict_dependent(x_true)[0]

        # create a PyMC model using the calibration model
        with pm.Model() as pmodel:
            x_hat = pm.Uniform("x_hat", 0, 1, shape=x_true.shape, transform=None)
            L = cmodel.loglikelihood(x=x_hat, y=y_obs, name='L_A01_A')
        assert isinstance(L, at.TensorVariable)
        assert L.ndim == 0

        # PyMC v4 returns the RV, but for .eval() we need the RV-value-variable
        if pm.__version__[0] != "3":
            x_hat = pmodel.rvs_to_values[x_hat]

        # compare the two loglikelihood computation methods
        x_test = numpy.random.normal(x_true, scale=0.1)
        actual = L.eval({
            x_hat: x_test
        })
        expected = cmodel.loglikelihood(x=x_test, y=y_obs)
        assert numpy.ndim(expected) == 0
        assert numpy.ndim(actual) == 0
        numpy.testing.assert_almost_equal(actual, expected, 6)
        pass

    @pytest.mark.parametrize("x", [
        numpy.array([1,2,3]),
        4,
    ])
    @pytest.mark.parametrize("y", [
        numpy.array([2,4,8]),
        5,
    ])
    def test_loglikelihood(self, x, y):
        cmodel = _TestPolynomialModel(independent_key='S', dependent_key='OD', mu_degree=1, scale_degree=1)
        cmodel.theta_fitted = [0, 1, 0.1, 1.6]

        actual = cmodel.loglikelihood(y=y, x=x)
        assert numpy.ndim(actual) == 0
        mu, scale, df = cmodel.predict_dependent(x, theta=cmodel.theta_fitted)
        expected = numpy.sum(stats.t.logpdf(x=y, loc=mu, scale=scale, df=df))
        numpy.testing.assert_equal(actual, expected)
        return

    def test_loglikelihood_exceptions(self):
        cmodel = _TestPolynomialModel(independent_key='S', dependent_key='OD', mu_degree=1, scale_degree=1)
        with pytest.raises(Exception, match="No parameter vector"):
            cmodel.loglikelihood(y=[2,3], x=[4,5])

        cmodel.theta_fitted = [0, 1, 0.1, 1.6]

        with pytest.raises(TypeError):
            cmodel.loglikelihood(4, x=[2,3])
        with pytest.raises(ValueError, match="Input x must be"):
            cmodel.loglikelihood(y=[2,3], x="hello")
        with pytest.raises(ValueError, match="operands could not be broadcast"):
            cmodel.loglikelihood(y=[1,2,3], x=[1,2])
        return

    def test_likelihood(self):
        # use a linear model with intercept 1 and slope 0.5
        cmodel = _TestPolynomialModel(independent_key="I", dependent_key="D", mu_degree=1)
        cmodel.theta_fitted = [1, 0.5, 0.5]

        assert numpy.isscalar(cmodel.likelihood(y=2, x=3))
        assert numpy.isscalar(cmodel.likelihood(y=[2, 3], x=[3, 4]))

        with pytest.raises(ValueError, match="operands could not be broadcast"):
            cmodel.likelihood(y=[1,2,3], x=[1,2])

        x_dense = numpy.linspace(0, 4, 501)
        actual = cmodel.likelihood(x=x_dense, y=2, scan_x=True)
        assert numpy.ndim(actual) == 1
        # the maximum likelihood should be at x=2
        assert x_dense[numpy.argmax(actual)] == 2
        pass


class TestBaseAsymmetricLogisticModelT:
    def test_predict_dependent(self):
        x = numpy.array([1,2,3])
        theta = [0, 4, 2, 1, 1, 0, 2, 1.4]
        cmodel = _TestLogisticModel(independent_key='S', dependent_key='OD', scale_degree=1)
        cmodel.theta_fitted = theta
        with pytest.raises(TypeError):
            _ = cmodel.predict_dependent(x, theta)
        mu, scale, df = cmodel.predict_dependent(x)
        numpy.testing.assert_array_equal(mu, calibr8.asymmetric_logistic(x, theta))
        numpy.testing.assert_array_equal(scale, 0 + 2 * mu)
        assert df == 1.4
        return
    
    def test_predict_independent(self):
        cmodel = _TestLogisticModel(independent_key='S', dependent_key='OD', scale_degree=1)
        cmodel.theta_fitted = [0, 4, 2, 1, 1, 2, 1.43]
        x_original = numpy.array([4, 5, 6])
        mu, sd, df = cmodel.predict_dependent(x_original)
        x_predicted = cmodel.predict_independent(y=mu)
        assert (numpy.allclose(x_predicted, x_original))
        return


class TestOptimization:
    def _get_test_model(self):
        theta_mu = (0.5, 1.4)
        theta_scale = (0.2,)
        theta = theta_mu + theta_scale + (4,)

        x = numpy.linspace(1, 10, 500)
        y = stats.t.rvs(
            loc=calibr8.polynomial(x, theta_mu),
            scale=calibr8.polynomial(x, theta_scale),
            df=theta[-1]
        )

        em = _TestPolynomialModel()
        return theta_mu, theta_scale, theta, em, x, y

    def test_finite_masking(self, caplog):
        x = numpy.random.normal(size=5)
        y = x ** 2
        result = calibr8.optimization._mask_and_warn_inf_or_nan(x, y)
        numpy.testing.assert_array_equal(result[0], x)
        numpy.testing.assert_array_equal(result[1], y)

        x[2] = float("nan")
        with caplog.at_level(logging.WARNING):
            result = calibr8.optimization._mask_and_warn_inf_or_nan(x, y, on="x")
        numpy.testing.assert_array_equal(result[0], x[~numpy.isnan(x)])
        numpy.testing.assert_array_equal(result[1], y[~numpy.isnan(x)])
        assert "1 elements" in caplog.text

        y[[0, 3]] = float("nan")
        with caplog.at_level(logging.WARNING):
            result = calibr8.optimization._mask_and_warn_inf_or_nan(x, y, on="y")
        numpy.testing.assert_array_equal(result[0], x[~numpy.isnan(y)])
        numpy.testing.assert_array_equal(result[1], y[~numpy.isnan(y)])
        assert "2 elements" in caplog.text

        with caplog.at_level(logging.WARNING):
            result = calibr8.optimization._mask_and_warn_inf_or_nan(x, y)
        assert not numpy.any(numpy.isnan(result[0]))
        assert not numpy.any(numpy.isnan(result[1]))
        assert "3 elements" in caplog.text
        pass

    def test_finite_masking_multivariate(self, caplog):
        x = numpy.random.normal(size=(4, 5))
        y = numpy.sum(x ** 2, axis=1)
        assert x.ndim == 2
        assert x.shape == (4, 5)
        assert y.shape == (4,)

        result = calibr8.optimization._mask_and_warn_inf_or_nan(x, y)
        numpy.testing.assert_array_equal(result[0], x)
        numpy.testing.assert_array_equal(result[1], y)

        x[1, 1] = numpy.inf
        with caplog.at_level(logging.WARNING):
            result = calibr8.optimization._mask_and_warn_inf_or_nan(x, y, on="x")
        assert len(result[0]) == 3
        assert len(result[1]) == 3
        assert "1 elements" in caplog.text

        y[[0, 3]] = numpy.nan
        with caplog.at_level(logging.WARNING):
            result = calibr8.optimization._mask_and_warn_inf_or_nan(x, y, on="y")
        assert len(result[0]) == 2
        assert len(result[1]) == 2
        assert "2 elements" in caplog.text

        with caplog.at_level(logging.WARNING):
            result = calibr8.optimization._mask_and_warn_inf_or_nan(x, y)
        assert len(result[0]) == 1
        assert len(result[1]) == 1
        assert "3 elements" in caplog.text

        # higher dimensionality inputs are not supported:
        with pytest.raises(ValueError, match="4-dimensional"):
            calibr8.optimization._mask_and_warn_inf_or_nan(x[:, :, None, None], y)
        pass

    def test_fit_checks_guess_and_bounds_count(self):
        theta_mu, theta_scale, theta, em, x, y = self._get_test_model()
        common = dict(model=em, independent=x, dependent=y)
        for fit in (calibr8.fit_scipy, calibr8.fit_pygmo):
            # wrong theta
            with pytest.raises(ValueError):
                fit(**common, theta_guess=numpy.ones(14), theta_bounds=[(-5, 5)]*len(theta))
            # wrong bounds
            with pytest.raises(ValueError):
                fit(**common, theta_guess=numpy.ones_like(theta), theta_bounds=[(-5, 5)]*14)
        return

    def test_fit_scipy(self, caplog):
        numpy.random.seed(1234)
        theta_mu, theta_scale, theta, em, x, y = self._get_test_model()
        theta_fit, history = calibr8.fit_scipy(
            em,
            independent=x, dependent=y,
            theta_guess=numpy.ones_like(theta),
            theta_bounds=[(-5, 5)]*len(theta_mu) + [(0.02, 1), (1, 20)]
        )
        for actual, desired, atol in zip(theta_fit, theta, [0.10, 0.05, 0.2, 2]):
            numpy.testing.assert_allclose(actual, desired, atol=atol)
        assert isinstance(history, list)
        numpy.testing.assert_array_equal(em.theta_fitted, theta_fit)
        assert em.theta_bounds is not None
        assert em.theta_guess is not None
        numpy.testing.assert_array_equal(em.cal_independent, x)
        numpy.testing.assert_array_equal(em.cal_dependent, y)

        with caplog.at_level(logging.WARNING):
            x[0] = float("nan")
            y[-1] = numpy.inf
            calibr8.fit_scipy(
                em,
                independent=x, dependent=y,
                theta_guess=numpy.ones_like(theta),
                theta_bounds=[(-5, 5)]*len(theta_mu) + [(0.02, 1), (1, 20)]
            )
        assert "2 elements" in caplog.text
        # inf/nan should only be ignored for fitting
        numpy.testing.assert_array_equal(em.cal_independent, x)
        numpy.testing.assert_array_equal(em.cal_dependent, y)
        pass

    @pytest.mark.xfail(reason="PyGMO and PyMC have dependencies (cloudpickle/dill) that are currently incompatible. See https://github.com/uqfoundation/dill/issues/383")
    @pytest.mark.skipif(not HAS_PYGMO, reason='requires PyGMO')
    def test_fit_pygmo(self, caplog):
        numpy.random.seed(1234)
        theta_mu, theta_scale, theta, em, x, y = self._get_test_model()
        theta_fit, history = calibr8.fit_pygmo(
            em,
            independent=x, dependent=y,
            theta_bounds=[(-5, 5)]*len(theta_mu) + [(0.02, 1), (1, 20)],
            evolutions=5,
        )
        for actual, desired, atol in zip(theta_fit, theta, [0.10, 0.05, 0.2, 2]):
            numpy.testing.assert_allclose(actual, desired, atol=atol)
        assert isinstance(history, list)
        numpy.testing.assert_array_equal(em.theta_fitted, theta_fit)
        assert em.theta_bounds is not None
        assert em.theta_guess is None
        numpy.testing.assert_array_equal(em.cal_independent, x)
        numpy.testing.assert_array_equal(em.cal_dependent, y)

        with caplog.at_level(logging.WARNING):
            x[0] = -numpy.inf
            y[-1] = float("nan")
            calibr8.fit_pygmo(
                em,
                independent=x, dependent=y,
                theta_bounds=[(-5, 5)]*len(theta_mu) + [(0.02, 1), (1, 20)],
                evolutions=5,
            )
        assert "2 elements" in caplog.text
        # inf/nan should only be ignored for fitting
        numpy.testing.assert_array_equal(em.cal_independent, x)
        numpy.testing.assert_array_equal(em.cal_dependent, y)
        pass
