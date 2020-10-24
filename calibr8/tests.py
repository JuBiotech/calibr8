import collections
import datetime
import numpy
import pathlib
import pytest
import scipy
import scipy.stats as stats

import calibr8


try:
    import pymc3
    import theano
    import theano.tensor as tt
    HAS_PYMC3 = True
except ModuleNotFoundError:
    HAS_PYMC3 = False

try:
    import pygmo
    HAS_PYGMO = True
except ModuleNotFoundError:
    HAS_PYGMO = False


dir_testfiles = pathlib.Path(pathlib.Path(__file__).absolute().parent, 'testfiles')


class _TestModel(calibr8.ErrorModel):
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


class TestBasicErrorModel:
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
        class EM_OK(calibr8.ErrorModel):
            def __init__(self, arg1=1, *, kwonly=2, kwonlydefault=4):
                super().__init__('I', 'D', theta_names=tuple('abc'))
        EM_OK()

        class EM_args(calibr8.ErrorModel):
            def __init__(self, arg1):
                super().__init__('I', 'D', theta_names=tuple('abc'))
        with pytest.raises(TypeError):
            EM_args(arg1=3)

        class EM_kwargs(calibr8.ErrorModel):
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
        errormodel = _TestModel()
        with pytest.raises(NotImplementedError):
            _ = errormodel.predict_dependent(x)
        with pytest.raises(NotImplementedError):
            _ = errormodel.predict_independent(x)
        with pytest.raises(NotImplementedError):
            _ = errormodel.infer_independent(y, lower=0, upper=10, steps=10, hdi_prob=None)
        with pytest.raises(NotImplementedError):
            _ = errormodel.loglikelihood(y=y, x=x, theta=[1,2,3])
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
        class DifferentEM(calibr8.ErrorModel):
            pass

        with pytest.raises(calibr8.CompatibilityException):
            DifferentEM.load('save_load_test.json')
        return

    def test_save_and_load_attributes(self):
        em = _TestModel()
        em.theta_guess = (1,1,1)
        em.theta_fitted = (1,2,3)
        theta_timestamp = em.theta_timestamp
        em.theta_bounds = (
            (None, None),
            (0, 5),
            (0, 10)
        )
        em.cal_independent = numpy.linspace(0, 10, 7)
        em.cal_dependent = numpy.random.normal(em.cal_independent)

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


@pytest.mark.skipif(not HAS_PYMC3, reason='requires PyMC3')
class TestSymbolicModelFunctions:
    def _check_numpy_theano_equivalence(self, function, theta):
        # make sure that test value computation is turned off (pymc3 likes to turn it on)
        with theano.configparser.change_flags(compute_test_value='off'):
            # create computation graph
            x = tt.vector('x', dtype=theano.config.floatX)
            y = function(x, theta)
            assert isinstance(y, tt.TensorVariable)

            # compile theano function
            f = theano.function([x], [y])
        
            # check equivalence of numpy and theano computation
            x_test = [1, 2, 4]
            numpy.testing.assert_almost_equal(
                f(x_test)[0],
                function(x_test, theta)
            )
        return

    def test_logistic(self):
        self._check_numpy_theano_equivalence(
            calibr8.logistic,
            [2, 2, 4, 1]
        )
        return

    def test_asymmetric_logistic(self):
        self._check_numpy_theano_equivalence(
            calibr8.asymmetric_logistic,
            [0, 4, 2, 1, 1]
        )
        return

    def test_log_log_logistic(self):
        self._check_numpy_theano_equivalence(
            calibr8.log_log_logistic,
            [2, 2, 4, 1]
        )
        return
    
    def test_xlog_logistic(self):
        self._check_numpy_theano_equivalence(
            calibr8.xlog_logistic,
            [2, 2, 4, 1]
        )
        return

    def test_ylog_logistic(self):
        self._check_numpy_theano_equivalence(
            calibr8.ylog_logistic,
            [2, 2, 4, 1]
        )
        return


class TestUtils:
    def test_datetime_parsing(self):
        assert calibr8.parse_datetime(None) is None
        assert (
            calibr8.parse_datetime('2018-12-01T09:27:30Z')
            ==
            datetime.datetime(2018, 12, 1, 9, 27, 30, tzinfo=datetime.timezone.utc)
        )
        assert (
            calibr8.parse_datetime('2018-12-01T09:27:30+0000')
            ==
            datetime.datetime(2018, 12, 1, 9, 27, 30, tzinfo=datetime.timezone.utc)
        )

    def test_datetime_formatting(self):
        assert calibr8.format_datetime(None) is None
        assert (
            calibr8.format_datetime(datetime.datetime(2018, 12, 1, 9, 27, 30, tzinfo=datetime.timezone.utc))
            ==
            '2018-12-01T09:27:30Z'
        )

    @pytest.mark.skipif(HAS_PYMC3, reason='run only if PyMC3 is not installed')
    def test_istensor_without_pymc3(self):
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

    @pytest.mark.skipif(not HAS_PYMC3, reason='requires PyMC3')
    def test_istensor_with_pymc3(self):
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
            'c': numpy.array([(1, tt.TensorVariable([1,2,3])), (3,4)])
        }
        assert (calibr8.istensor(test_dict2))
        assert (calibr8.istensor([1, tt.as_tensor_variable([1,2]), 3]))
        assert (calibr8.istensor([1, tt.TensorVariable([1,2]), 3]))
        assert (calibr8.istensor(numpy.array([1, tt.TensorVariable([1,2]), 3])))

    def test_import_warner(self):
        dummy = calibr8.utils.ImportWarner('dummy')
        with pytest.raises(ImportError):
            print(dummy.__version__)
        return

    @pytest.mark.skipif(HAS_PYMC3, reason='run only if PyMC3 is not installed')
    def test_has_modules(self):
        assert not (calibr8.utils.HAS_THEANO)
        assert not (calibr8.utils.HAS_PYMC3)
        return

    @pytest.mark.skipif(not HAS_PYMC3, reason='requires PyMC3')
    def test_has_modules(self):
        assert (calibr8.utils.HAS_THEANO)
        assert (calibr8.utils.HAS_PYMC3)
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

    def test_guess_asymmetric_logistic_theta(self):
        with pytest.raises(ValueError):
            calibr8.guess_asymmetric_logistic_theta([1,2,3], [1,2])
        with pytest.raises(ValueError):
            calibr8.guess_asymmetric_logistic_theta([1,2], [[1,2],[2,3]])
        L_L, L_U, I_x, S, c = calibr8.guess_asymmetric_logistic_theta(
            X=[0, 1, 2, 3, 4, 5],
            Y=[0, 1, 2, 3, 4, 5],
        )
        assert L_L == 0
        assert L_U == 10
        assert I_x == (0+5)/2
        numpy.testing.assert_almost_equal(S, 1)
        assert c == -1
        return

    def test_guess_asymmetric_logistic_bounds(self):
        with pytest.raises(ValueError):
            calibr8.guess_asymmetric_logistic_theta([1,2,3], [1,2])
        with pytest.raises(ValueError):
            calibr8.guess_asymmetric_logistic_theta([1,2], [[1,2],[2,3]])
        
        for half_open in (True, False):
            L_L, L_U, I_x, S, c = calibr8.guess_asymmetric_logistic_bounds(
                X=[0, 1, 2, 3, 4, 5],
                Y=[0, 1, 2, 3, 4, 5],
                half_open=half_open
            )
            if half_open:
                numpy.testing.assert_allclose(L_L, (-numpy.inf, 2.5))
                numpy.testing.assert_allclose(L_U, (2.5, numpy.inf))
            else:
                numpy.testing.assert_allclose(L_L, (0-100*5, 2.5))
                numpy.testing.assert_allclose(L_U, (2.5, 5+100*5))
            numpy.testing.assert_allclose(I_x, (-15, 20))
            numpy.testing.assert_allclose(S, (0, 10))
            numpy.testing.assert_allclose(c, (-5, 5))
        return

    def test_guess_asymmetric_logistic_theta_in_bounds(self):
        X = numpy.linspace(0, 50, 42)
        Y = numpy.random.normal(X, scale=0.4)
        theta = calibr8.guess_asymmetric_logistic_theta(X, Y)
        for half_open in (True, False):
            bounds = calibr8.guess_asymmetric_logistic_bounds(X, Y, half_open=half_open)
            for t, (lb, ub) in zip(theta, bounds):
                assert t > lb
                assert t < ub
        return


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
        x, pdf, median, hdi_prob, lower_x, upper_x = em.infer_independent(y=1, lower=0, upper=20, steps=876)


        assert len(x) == len(pdf)
        assert x[0] == 0
        assert x[-1] == 20
        assert (numpy.isclose(scipy.integrate.cumtrapz(pdf,x)[-1], 1, atol=0.0001))
        assert (lower_x==0)
        assert (upper_x==20)
        assert (hdi_prob==1)

        # check trimming to [2.5,97.5] interval
        posterior = em.infer_independent(y=1, lower=0, upper=20, steps=1775, hdi_prob=0.95)

        assert (len(posterior)==6)
        assert len(posterior.x_dense) == len(posterior.pdf)
        assert (posterior.hdi_prob==0.95)
        assert (numpy.isclose(scipy.integrate.cumtrapz(posterior.pdf,posterior.x_dense)[-1], 0.95, atol=0.0001))

        # check that error are raised by wrong input
        with pytest.raises(ValueError):
            _ = em.infer_independent(y=1, lower=0, upper=20, steps=1000, hdi_prob=(-1))
        with pytest.raises(ValueError):
            _ = em.infer_independent(y=1, lower=0, upper=20, steps=1000, hdi_prob=(97.5))
        pass
    
    @pytest.mark.skipif(not HAS_PYMC3, reason='requires PyMC3')
    def test_symbolic_loglikelihood(self):
        errormodel = _TestPolynomialModel(independent_key='S', dependent_key='A', mu_degree=1, scale_degree=1)
        errormodel.theta_fitted = [0, 1, 0.1, 1]
       
        # create test data
        x_true = numpy.array([1,2,3,4,5])
        y_obs = errormodel.predict_dependent(x_true)[0]

        # create a pymc3 model using the error model
        with pymc3.Model() as pmodel:
            x_hat = pymc3.Uniform('x_hat', lower=0, upper=10, shape=x_true.shape, transform=None)
            L = errormodel.loglikelihood(x=x_hat, y=y_obs, replicate_id='A01', dependent_key='A')
            assert isinstance(L, tt.TensorVariable)
        
        # compare the two loglikelihood computation methods
        x_test = numpy.random.normal(x_true, scale=0.1)
        actual = L.logp({
            'x_hat': x_test
        })
        expected = errormodel.loglikelihood(x=x_test, y=y_obs)
        numpy.testing.assert_almost_equal(actual, expected, 6)
        pass

    def test_loglikelihood(self):
        x = numpy.array([1,2,3])
        y = numpy.array([1,2,3])
        errormodel = _TestPolynomialModel(independent_key='S', dependent_key='OD', mu_degree=1, scale_degree=1)
        errormodel.theta_fitted = [0, 1, 0.1, 1.6]
        with pytest.raises(TypeError):
            _ = errormodel.loglikelihood(y, x=x)
        true = errormodel.loglikelihood(y=y, x=x)
        mu, scale, df = errormodel.predict_dependent(x, theta=errormodel.theta_fitted)
        expected = numpy.sum(stats.t.logpdf(x=y, loc=mu, scale=scale, df=df))
        assert expected == true
        x = 'hello'
        with pytest.raises(Exception):
            _= errormodel.loglikelihood(y=y, x=x)
        return
    
    def test_loglikelihood_without_fit(self):
        x = numpy.array([1,2,3])
        y = numpy.array([1,2,3])
        errormodel = _TestPolynomialModel(independent_key='Glu', dependent_key='OD', mu_degree=1, scale_degree=1)
        with pytest.raises(Exception):
            _= errormodel.loglikelihood(y=y, x=x)
        return


class TestBaseAsymmetricLogisticModelT:
    def test_predict_dependent(self):
        x = numpy.array([1,2,3])
        theta = [0, 4, 2, 1, 1, 0, 2, 1.4]
        errormodel = _TestLogisticModel(independent_key='S', dependent_key='OD', scale_degree=1)
        errormodel.theta_fitted = theta
        with pytest.raises(TypeError):
            _ = errormodel.predict_dependent(x, theta)
        mu, scale, df = errormodel.predict_dependent(x)
        numpy.testing.assert_array_equal(mu, calibr8.asymmetric_logistic(x, theta))
        numpy.testing.assert_array_equal(scale, 0 + 2 * mu)
        assert df == 1.4
        return
    
    def test_predict_independent(self):
        errormodel = _TestLogisticModel(independent_key='S', dependent_key='OD', scale_degree=1)
        errormodel.theta_fitted = [0, 4, 2, 1, 1, 2, 1.43]
        x_original = numpy.array([4, 5, 6])
        mu, sd, df = errormodel.predict_dependent(x_original)
        x_predicted = errormodel.predict_independent(y=mu)
        assert (numpy.allclose(x_predicted, x_original))
        return
    
    @pytest.mark.skipif(not HAS_PYMC3, reason='requires PyMC3')
    def test_symbolic_loglikelihood(self):
        errormodel = _TestLogisticModel(independent_key='S', dependent_key='A', scale_degree=1)
        errormodel.theta_fitted = [0, 4, 2, 1, 1, 2, 1.23]
       
        # create test data
        x_true = numpy.array([1,2,3,4,5])
        y_obs = errormodel.predict_dependent(x_true)[0]

        # create a pymc3 model using the error model
        with pymc3.Model() as pmodel:
            x_hat = pymc3.Uniform('x_hat', lower=0, upper=10, shape=x_true.shape, transform=None)
            L = errormodel.loglikelihood(x=x_hat, y=y_obs, replicate_id='A01', dependent_key='A')
            assert isinstance(L, tt.TensorVariable)
        
        # compare the two loglikelihood computation methods
        x_test = numpy.random.normal(x_true, scale=0.1)
        actual = L.logp({
            'x_hat': x_test
        })
        expected = errormodel.loglikelihood(x=x_test, y=y_obs)
        numpy.testing.assert_almost_equal(actual, expected, 6)
        return
    
    def test_loglikelihood(self):
        x = numpy.array([1,2,3])
        y = numpy.array([1,2,3])
        errormodel = _TestLogisticModel(independent_key='S', dependent_key='OD', scale_degree=1)
        errormodel.theta_fitted = [0, 4, 2, 1, 1, 2, 1.7]
        with pytest.raises(TypeError):
            _ = errormodel.loglikelihood(y, x=x)
        true = errormodel.loglikelihood(y=y, x=x)
        mu, scale, df = errormodel.predict_dependent(x, theta=errormodel.theta_fitted)
        expected = numpy.sum(stats.t.logpdf(x=y, loc=mu, scale=scale, df=df))
        assert expected == true
        x = 'hello'
        with pytest.raises(Exception):
            _= errormodel.loglikelihood(y=y, x=x)
        return
    
    def test_loglikelihood_without_fit(self):
        x = numpy.array([1,2,3])
        y = numpy.array([1,2,3])
        errormodel = _TestLogisticModel(independent_key='Glu', dependent_key='OD', scale_degree=1)
        with pytest.raises(Exception):
            _= errormodel.loglikelihood(y=y, x=x)
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

    def test_fit_scipy(self):
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
        pass

    @pytest.mark.skipif(not HAS_PYGMO, reason='requires PyGMO')
    def test_fit_pygmo(self):
        numpy.random.seed(1234)
        theta_mu, theta_scale, theta, em, x, y = self._get_test_model()
        theta_fit, history = calibr8.fit_pygmo(
            em,
            independent=x, dependent=y,
            theta_bounds=[(-5, 5)]*len(theta_mu) + [(0.02, 1), (1, 20)]
        )
        for actual, desired, atol in zip(theta_fit, theta, [0.10, 0.05, 0.2, 2]):
            numpy.testing.assert_allclose(actual, desired, atol=atol)
        assert isinstance(history, list)
        numpy.testing.assert_array_equal(em.theta_fitted, theta_fit)
        assert em.theta_bounds is not None
        assert em.theta_guess is None
        numpy.testing.assert_array_equal(em.cal_independent, x)
        numpy.testing.assert_array_equal(em.cal_dependent, y)
        pass
