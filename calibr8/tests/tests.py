import collections
import unittest
import numpy
import pathlib
import scipy.stats as stats

import calibr8


try:
    import pymc3
    import theano
    import theano.tensor as tt
    HAS_PYMC3 = True
except ModuleNotFoundError:
    HAS_PYMC3 = False



dir_testfiles = pathlib.Path(pathlib.Path(__file__).absolute().parent, 'testfiles')
       

class ErrorModelTest(unittest.TestCase):
    def test_init(self):
        independent = 'X'
        dependent = 'BS'
        errormodel = calibr8.ErrorModel(independent, dependent)
        self.assertEqual(errormodel.independent_key, independent)
        self.assertEqual(errormodel.dependent_key, dependent)
        self.assertEqual(errormodel.theta_fitted, None)
    
    def test_exceptions(self):
        independent = 'X'
        dependent = 'BS'
        x = numpy.array([1,2,3])
        y = numpy.array([4,5,6])
        errormodel = calibr8.ErrorModel(independent, dependent)
        with self.assertRaises(NotImplementedError):
            _ = errormodel.predict_dependent(x)
        with self.assertRaises(NotImplementedError):
            _ = errormodel.predict_independent(x)
        with self.assertRaises(NotImplementedError):
            _ = errormodel.infer_independent(y)
        with self.assertRaises(NotImplementedError):
            _ = errormodel.loglikelihood(y=y, x=x, theta=[1,2,3])
        with self.assertRaises(NotImplementedError):
            _ = errormodel.fit(independent=x, dependent=y, theta_guessed=None)
        return
    

class TestModelFunctions(unittest.TestCase):
    def test_logistic(self):
        x = numpy.array([1.,2.,4.])
        theta = [2,2,4,1]
        expected = 2*2-4+(2*(4-2))/(1+numpy.exp(-2*1/(4-2)*(x-2)))
        true = calibr8.logistic(x, theta)
        self.assertTrue(numpy.array_equal(true, expected))
        return

    def test_inverse_logistic(self):
        x = numpy.array([1.,2.,4.])
        theta = [2,2,4,1]
        forward = calibr8.logistic(x, theta)
        reverse = calibr8.inverse_logistic(forward, theta)
        self.assertTrue(numpy.allclose(x, reverse))
        return

    def test_asymmetric_logistic(self):
        x = numpy.array([1.,2.,4.])
        theta = [0,4,2,1,1]
        expected = 0+(4-0)/(1+numpy.exp(-1*(x-2)))
        true = calibr8.asymmetric_logistic(x, theta)
        self.assertTrue(numpy.array_equal(true, expected))
        expected = calibr8.logistic(x, theta=[2,2,4,1])
        self.assertTrue(numpy.array_equal(true, expected))
        return

    def test_inverse_asymmetric_logistic(self):
        x = numpy.array([1.,2.,4.])
        theta = [0,4,2,1,1]
        forward = calibr8.asymmetric_logistic(x, theta)
        reverse = calibr8.inverse_asymmetric_logistic(forward, theta)
        self.assertTrue(numpy.allclose(x, reverse))
        return 

    def test_log_log_logistic(self):
        x = numpy.array([1.,2.,4.])
        theta = [2,2,4,1]
        expected = numpy.exp(2*2-4+(2*(4-2))/(1+numpy.exp(-2*1/(4-2)*(numpy.log(x)-2))))
        true = calibr8.log_log_logistic(x, theta)
        self.assertTrue(numpy.array_equal(true, expected))
        expected = numpy.exp(calibr8.logistic(numpy.log(x), theta))   
        self.assertTrue(numpy.array_equal(true, expected))
        return

    def test_inverse_log_log_logistic(self):
        x = numpy.array([1.,2.,4.])
        theta = [2,2,4,1]
        forward = calibr8.log_log_logistic(x, theta)
        reverse = calibr8.inverse_log_log_logistic(forward, theta)
        self.assertTrue(numpy.allclose(x, reverse))
        return

    def test_xlog_logistic(self):
        x = numpy.array([1.,2.,4.])
        theta = [2,2,4,1]
        expected = 2*2-4+(2*(4-2))/(1+numpy.exp(-2*1/(4-2)*(numpy.log(x)-2)))
        true = calibr8.xlog_logistic(x, theta)
        self.assertTrue(numpy.array_equal(true, expected))
        expected = calibr8.logistic(numpy.log(x), theta)
        self.assertTrue(numpy.array_equal(true, expected))        
        return
        
    def test_inverse_xlog_logistic(self):
        x = numpy.array([1.,2.,4.])
        theta = [2,2,4,1]
        forward = calibr8.xlog_logistic(x, theta)
        reverse = calibr8.inverse_xlog_logistic(forward, theta)
        self.assertTrue(numpy.allclose(x, reverse))
        return

    def test_ylog_logistic(self):
        x = numpy.array([1.,2.,4.])
        theta = [2,2,4,1]
        expected = numpy.exp(2*2-4+(2*(4-2))/(1+numpy.exp(-2*1/(4-2)*(x-2))))
        true = calibr8.ylog_logistic(x, theta)
        self.assertTrue(numpy.array_equal(true, expected))
        expected = numpy.exp(calibr8.logistic(x, theta))
        self.assertTrue(numpy.array_equal(true, expected))
        return

    def test_inverse_ylog_logistic(self):
        x = numpy.array([1.,2.,4.])
        theta = [2,2,4,1]
        forward = calibr8.ylog_logistic(x, theta)
        reverse = calibr8.inverse_ylog_logistic(forward, theta)
        self.assertTrue(numpy.allclose(x, reverse))
        return


@unittest.skipUnless(HAS_PYMC3, 'requires PyMC3')
class TestSymbolicModelFunctions(unittest.TestCase):
    def _check_numpy_theano_equivalence(self, function, theta):
        # make sure that test value computation is turned off (pymc3 likes to turn it on)
        with theano.configparser.change_flags(compute_test_value='off'):
            # create computation graph
            x = tt.vector('x', dtype=theano.config.floatX)
            y = function(x, theta)
            self.assertIsInstance(y, tt.TensorVariable)

            # compile theano function
            f = theano.function([x], [y])
        
            # check equivalence of numpy and theano computation
            x_test = [1, 2, 4]
            self.assertTrue(numpy.array_equal(
                f(x_test)[0],
                function(x_test, theta)
            ))
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
        
        
class UtilsTest(unittest.TestCase):
    @unittest.skipIf(HAS_PYMC3, "only if PyMC3 is not imported")
    def test_istensor_without_pymc3(self):
        test_dict = {
            'a': 1, 
            'b': [1,2,3], 
            'c': numpy.array([(1,2), (3,4)])
            }
        self.assertFalse(calibr8.istensor(test_dict))
        self.assertFalse(calibr8.istensor(1.2))
        self.assertFalse(calibr8.istensor(-5))
        self.assertFalse(calibr8.istensor([1,2,3]))
        self.assertFalse(calibr8.istensor('hello'))

    @unittest.skipUnless(HAS_PYMC3, 'requires PyMC3')
    def test_istensor_with_pymc3(self):
        test_dict = {
            'a': 1, 
            'b': [1,2,3], 
            'c': numpy.array([(1,2), (3,4)])
            }
        self.assertFalse(calibr8.istensor(test_dict))
        self.assertFalse(calibr8.istensor(1.2))
        self.assertFalse(calibr8.istensor(-5))
        self.assertFalse(calibr8.istensor([1,2,3]))
        self.assertFalse(calibr8.istensor('hello'))
            
        test_dict2 = {
            'a': 1, 
            'b': [1,2,3], 
            'c': numpy.array([(1, tt.TensorVariable([1,2,3])), (3,4)])
            }
        self.assertTrue(calibr8.istensor(test_dict2))
        self.assertTrue(calibr8.istensor([1, tt.TensorVariable([1,2]), 3]))
        self.assertTrue(calibr8.istensor(numpy.array([1, tt.TensorVariable([1,2]), 3])))


class BaseGlucoseErrorModelTest(unittest.TestCase):
    def test_errors(self):
        independent = 'S'
        dependent = 'OD'
        y = numpy.array([1,2,3])
        x = numpy.array([1,2,3])
        theta = [0,0,0]
        errormodel = calibr8.BaseGlucoseErrorModel(independent, dependent)
        errormodel.theta_fitted = [0,1,0.1]
        with self.assertRaises(NotImplementedError):
            _ = errormodel.predict_dependent(x)
        with self.assertRaises(NotImplementedError):
            _ = errormodel.infer_independent(y)
        with self.assertRaises(NotImplementedError):
            _ = errormodel.loglikelihood(y=y, x=x)
        with self.assertRaises(NotImplementedError):
            _ = errormodel.fit(independent=x, dependent=y, theta_guessed=None, bounds=None)
        return


class LinearGlucoseErrorModelTest(unittest.TestCase):
    def test_predict_dependent(self):
        independent = 'S'
        dependent = 'OD'
        x = numpy.array([1,2,3])
        theta = [0,0,0]
        errormodel = calibr8.LinearGlucoseErrorModel(independent, dependent)
        errormodel.theta_fitted = [0,1,0.1]
        with self.assertRaises(TypeError):
            _ = errormodel.predict_dependent(x, theta)
        mu, sigma, df = errormodel.predict_dependent(x)
        self.assertTrue(numpy.array_equal(mu, numpy.array([1,2,3])))
        self.assertTrue(numpy.array_equal(sigma, numpy.array([0.1,0.1,0.1])))
        self.assertEqual(df, 1)
        return
    
    def test_predict_independent(self):
        errormodel = calibr8.LinearGlucoseErrorModel('S', 'OD')
        errormodel.theta_fitted = [0, 2, 0.1]
        x_original = numpy.array([4, 5, 6])
        mu, sd, df = errormodel.predict_dependent(x_original)
        x_predicted = errormodel.predict_independent(y=mu)
        self.assertTrue(numpy.array_equal(mu, [8, 10, 12]))
        self.assertTrue(numpy.array_equal(sd, [0.1, 0.1, 0.1]))
        self.assertTrue(numpy.allclose(x_predicted, x_original))
        return
    
    @unittest.skipUnless(HAS_PYMC3, "requires PyMC3")
    def test_infer_independent(self):
        errormodel = calibr8.LinearGlucoseErrorModel('S', 'OD')
        errormodel.theta_fitted = [0, 2, 0.1]
        trace = errormodel.infer_independent(y=1, draws=1)
        self.assertTrue(len(trace)==1)
        self.assertTrue(len(trace['Glucose'][0]==1))
        return

    @unittest.skipIf(HAS_PYMC3, "only if PyMC3 is not imported")
    def test_error_infer_independent(self):
        errormodel = calibr8.LinearGlucoseErrorModel('S', 'OD')
        with self.assertRaises(ImportError):
            _ = errormodel.infer_independent(y=1, draws=1)
        return

    @unittest.skipUnless(HAS_PYMC3, "requires PyMC3")
    def test_symbolic_loglikelihood(self):
        errormodel = calibr8.LinearGlucoseErrorModel('S', 'A')
        errormodel.theta_fitted = [0,1,0.1]
       
        # create test data
        x_true = numpy.array([1,2,3,4,5])
        y_obs = errormodel.predict_dependent(x_true)[0]

        # create a pymc3 model using the error model
        with pymc3.Model() as pmodel:
            x_hat = pymc3.Uniform('x_hat', lower=0, upper=10, shape=x_true.shape, transform=None)
            L = errormodel.loglikelihood(x=x_hat, y=y_obs, replicate_id='A01', dependent_key='A')
            self.assertIsInstance(L, tt.TensorVariable)
        
        # compare the two loglikelihood computation methods
        x_test = numpy.random.normal(x_true, scale=0.1)
        actual = L.logp({
            'x_hat': x_test
        })
        expected = errormodel.loglikelihood(x=x_test, y=y_obs)
        self.assertAlmostEqual(actual, expected, 6)
        return

    def test_loglikelihood(self):
        independent = 'S'
        dependent = 'OD'
        x = numpy.array([1,2,3])
        y = numpy.array([1,2,3])
        errormodel = calibr8.LinearGlucoseErrorModel(independent, dependent)
        errormodel.theta_fitted = [0,1,0.1]
        with self.assertRaises(TypeError):
            _ = errormodel.loglikelihood(y, x=x)
        true = errormodel.loglikelihood(y=y, x=x)
        mu, sigma, df = errormodel.predict_dependent(x, theta=errormodel.theta_fitted)
        expected = numpy.sum(numpy.log(stats.t.pdf(x=y, loc=mu, scale=sigma, df=1)))
        self.assertEqual(expected, true)
        x = 'hello'
        with self.assertRaises(Exception):
            _= errormodel.loglikelihood(y=y, x=x)
        return
    
    def test_loglikelihood_without_fit(self):
        independent = 'Glu'
        dependent = 'OD'
        x = numpy.array([1,2,3])
        y = numpy.array([1,2,3])
        errormodel = calibr8.LinearGlucoseErrorModel(independent, dependent)
        with self.assertRaises(Exception):
            _= errormodel.loglikelihood(y=y, x=x)
        return


class LogisticGlucoseErrorModelTest(unittest.TestCase):
    def test_predict_dependent(self):
        independent = 'S'
        dependent = 'OD'
        x = numpy.array([1,2,3])
        theta = [0,4,2,1,1,0,2]
        errormodel = calibr8.LogisticGlucoseErrorModel(independent, dependent)
        errormodel.theta_fitted = theta
        with self.assertRaises(TypeError):
            _ = errormodel.predict_dependent(x, theta)
        mu, sigma, df = errormodel.predict_dependent(x)
        self.assertTrue(numpy.array_equal(mu, calibr8.asymmetric_logistic(x, theta)))
        self.assertTrue(numpy.array_equal(sigma, 2*mu))
        self.assertEqual(df, 1)
        return
    
    def test_predict_independent(self):
        errormodel = calibr8.LogisticGlucoseErrorModel('S', 'OD')
        errormodel.theta_fitted = [0,4,2,1,1,2,0]
        x_original = numpy.array([4, 5, 6])
        mu, sd, df = errormodel.predict_dependent(x_original)
        x_predicted = errormodel.predict_independent(y=mu)
        self.assertTrue(numpy.allclose(x_predicted, x_original))
        return
    
    @unittest.skipUnless(HAS_PYMC3, "requires PyMC3")
    def test_infer_independent(self):
        errormodel = calibr8.LogisticGlucoseErrorModel('S', 'OD')
        errormodel.theta_fitted = [0,4,2,1,1,2,0]
        trace = errormodel.infer_independent(y=1, draws=1)
        self.assertTrue(len(trace)==1)
        self.assertTrue(len(trace['Glucose'][0]==1))
        return

    @unittest.skipIf(HAS_PYMC3, "only if PyMC3 is not imported")
    def test_error_infer_independent(self):
        errormodel = calibr8.LogisticGlucoseErrorModel('S', 'OD')
        with self.assertRaises(ImportError):
            _ = errormodel.infer_independent(y=1, draws=1)
        return

    @unittest.skipUnless(HAS_PYMC3, "requires PyMC3")
    def test_symbolic_loglikelihood(self):
        errormodel = calibr8.LogisticGlucoseErrorModel('S', 'A')
        errormodel.theta_fitted = [0,4,2,1,1,2,0]
       
        # create test data
        x_true = numpy.array([1,2,3,4,5])
        y_obs = errormodel.predict_dependent(x_true)[0]

        # create a pymc3 model using the error model
        with pymc3.Model() as pmodel:
            x_hat = pymc3.Uniform('x_hat', lower=0, upper=10, shape=x_true.shape, transform=None)
            L = errormodel.loglikelihood(x=x_hat, y=y_obs, replicate_id='A01', dependent_key='A')
            self.assertIsInstance(L, tt.TensorVariable)
        
        # compare the two loglikelihood computation methods
        x_test = numpy.random.normal(x_true, scale=0.1)
        actual = L.logp({
            'x_hat': x_test
        })
        expected = errormodel.loglikelihood(x=x_test, y=y_obs)
        self.assertAlmostEqual(actual, expected, 6)
        return
    
    def test_loglikelihood(self):
        independent = 'S'
        dependent = 'OD'
        x = numpy.array([1,2,3])
        y = numpy.array([1,2,3])
        errormodel = calibr8.LogisticGlucoseErrorModel(independent, dependent)
        errormodel.theta_fitted = [0,4,2,1,1,2,0]
        with self.assertRaises(TypeError):
            _ = errormodel.loglikelihood(y, x=x)
        true = errormodel.loglikelihood(y=y, x=x)
        mu, sigma, df = errormodel.predict_dependent(x, theta=errormodel.theta_fitted)
        expected = numpy.sum(numpy.log(stats.t.pdf(x=y, loc=mu, scale=sigma, df=df)))
        self.assertEqual(expected, true)
        x = 'hello'
        with self.assertRaises(Exception):
            _= errormodel.loglikelihood(y=y, x=x)
        return
    
    def test_loglikelihood_without_fit(self):
        independent = 'Glu'
        dependent = 'OD'
        x = numpy.array([1,2,3])
        y = numpy.array([1,2,3])
        errormodel = calibr8.LogisticGlucoseErrorModel(independent, dependent)
        with self.assertRaises(Exception):
            _= errormodel.loglikelihood(y=y, x=x)
        return


class BiomassErrorModelTest(unittest.TestCase):
    def test_predict_dependent(self):
        independent = 'BTM'
        dependent = 'BS'
        x = numpy.array([1,2,3])
        theta = [0,4,2,1,1,0,2]
        errormodel = calibr8.BiomassErrorModel(independent, dependent)
        errormodel.theta_fitted = theta
        with self.assertRaises(TypeError):
            _ = errormodel.predict_dependent(x, theta)
        mu, sigma, df = errormodel.predict_dependent(x)
        self.assertTrue(numpy.array_equal(mu, calibr8.asymmetric_logistic(x, theta)))
        self.assertTrue(numpy.array_equal(sigma, 2*mu))        
        self.assertEqual(df, 1)
        return

    def test_predict_independent(self):
        errormodel = calibr8.BiomassErrorModel('X', 'BS')
        errormodel.theta_fitted = numpy.array([0,4,2,1,1,2,0])
        x_original = numpy.linspace(0.01, 30, 20)
        mu, sd, df = errormodel.predict_dependent(x_original)
        x_predicted = errormodel.predict_independent(y=mu)
        self.assertTrue(numpy.allclose(x_predicted, x_original))
        return

    @unittest.skipUnless(HAS_PYMC3, "requires PyMC3")
    def test_infer_independent(self):
        errormodel = calibr8.BiomassErrorModel('X', 'BS')
        errormodel.theta_fitted = numpy.array([0,4,2,1,1,2,0])
        trace = errormodel.infer_independent(y=1, draws=1)
        self.assertTrue(len(trace)==1)
        self.assertTrue(len(trace['CDW'][0]==1))
        return

    @unittest.skipIf(HAS_PYMC3, "only if PyMC3 is not imported")
    def test_error_infer_independent(self):
        errormodel = calibr8.BiomassErrorModel('X', 'BS')
        with self.assertRaises(ImportError):
            errormodel.infer_independent(1)
        return

    @unittest.skipUnless(HAS_PYMC3, "requires PyMC3")
    def test_symbolic_loglikelihood(self):
        errormodel = calibr8.BiomassErrorModel('X', 'BS')
        errormodel.theta_fitted = numpy.array([0,4,2,1,1,2,0])

        # create test data
        x_true = numpy.array([1,2,3,4,5])
        y_obs = errormodel.predict_dependent(x_true)[0]

        # create a pymc3 model using the error model
        with pymc3.Model() as pmodel:
            x_hat = pymc3.Uniform('x_hat', lower=0, upper=10, shape=x_true.shape, transform=None)
            L = errormodel.loglikelihood(x=x_hat, y=y_obs, replicate_id='A01', dependent_key='BS')
            self.assertIsInstance(L, tt.TensorVariable)
        
        # compare the two loglikelihood computation methods
        x_test = numpy.random.normal(x_true, scale=0.1)
        actual = L.logp({
            'x_hat': x_test
        })
        expected = errormodel.loglikelihood(x=x_test, y=y_obs)
        self.assertAlmostEqual(actual, expected, 6)
        return

    def test_loglikelihood(self):
        independent = 'X'
        dependent = 'BS'
        x = numpy.array([1,2,3])
        y = numpy.array([1,2,3])
        errormodel = calibr8.BiomassErrorModel(independent, dependent)
        errormodel.theta_fitted = numpy.array([0,4,2,1,1,2,0])
        with self.assertRaises(TypeError):
            _ = errormodel.loglikelihood(y, x=x, theta=errormodel.theta_fitted)
        theta = errormodel.theta_fitted
        true = errormodel.loglikelihood(y=y, x=x, theta=theta)
        mu, sigma, df = errormodel.predict_dependent(x, theta=theta)
        expected = numpy.sum(numpy.log(stats.t.pdf(x=y, loc=mu, scale=sigma, df=1)))
        self.assertEqual(expected, true)
        x = 'hello'
        with self.assertRaises(Exception):
            _= errormodel.loglikelihood(y=y, x=x)
        return
    
    def test_loglikelihood_without_fit(self):
        independent = 'X'
        dependent = 'BS'
        x = numpy.array([1,2,3])
        y = numpy.array([1,2,3])
        errormodel = calibr8.BiomassErrorModel(independent, dependent)
        with self.assertRaises(Exception):
            _= errormodel.loglikelihood(y=y, x=x)
        return


class TestSerialDilutionPlanning(unittest.TestCase):
    def test_argchecking(self):
        with self.assertRaises(ValueError):
            calibr8.utils.DilutionPlan(
                xmin=0.001, xmax=30,
                R=8, C=12,
                stock=20,
                mode='log',
                vmax=1000,
                min_transfer=20
            )

        with self.assertRaises(ValueError):
            calibr8.utils.DilutionPlan(
                xmin=0.001, xmax=30,
                R=8, C=12,
                stock=30,
                mode='invalid',
                vmax=1000,
                min_transfer=20
            )

        with self.assertRaises(ValueError):
            calibr8.utils.DilutionPlan(
                xmin=0.001, xmax=30,
                R=6, C=4,
                stock=30,
                mode='linear',
                vmax=1000,
                min_transfer=20
            )

        return

    def test_repr(self):
        plan = calibr8.utils.DilutionPlan(
            xmin=0.001, xmax=30,
            R=8, C=12,
            stock=30,
            mode='log',
            vmax=1000,
            min_transfer=20
        )

        out = plan.__repr__()

        self.assertIsNotNone(out)
        self.assertIsInstance(out, str)
        return

    def test_linear_plan(self):
        plan = calibr8.utils.DilutionPlan(
            xmin=1, xmax=10,
            R=10, C=1,
            stock=20,
            mode='linear',
            vmax=1000,
            min_transfer=20
        )

        self.assertTrue(numpy.array_equal(plan.x, plan.ideal_x))
        self.assertEqual(plan.max_steps, 0)
        self.assertEqual(plan.v_stock, 2750)
        self.assertEqual(plan.instructions[0][0], 0)
        self.assertEqual(plan.instructions[0][1], 0)
        self.assertEqual(plan.instructions[0][2], 'stock')
        self.assertTrue(numpy.array_equal(plan.instructions[0][3], [500, 450, 400, 350, 300, 250, 200, 150, 100,  50,]))
        return

    def test_log_plan(self):
        plan = calibr8.utils.DilutionPlan(
            xmin=0.01, xmax=10,
            R=4, C=3,
            stock=20,
            mode='log',
            vmax=1000,
            min_transfer=20
        )

        self.assertTrue(numpy.allclose(plan.x, plan.ideal_x, rtol=0.05))
        self.assertEqual(plan.max_steps, 2)
        self.assertEqual(plan.v_stock, 985)
        self.assertEqual(plan.instructions[0][0], 0)
        self.assertEqual(plan.instructions[0][1], 0)
        self.assertEqual(plan.instructions[0][2], 'stock')
        self.assertTrue(numpy.array_equal(plan.instructions[0][3], [500, 267, 142, 76]))
        self.assertTrue(numpy.array_equal(plan.instructions[1][3], [82, 82, 82, 82]))
        self.assertTrue(numpy.array_equal(plan.instructions[2][3], [81, 81, 81, 81]))
        return


if __name__ == '__main__':
    unittest.main(exit=False)
