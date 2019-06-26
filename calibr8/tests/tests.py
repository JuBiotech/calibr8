import collections
import unittest
import numpy
import pathlib
import scipy.stats as stats

import calibr8


try:
    import pymc3
    HAVE_PYMC3 = True
except ModuleNotFoundError:
    HAVE_PYMC3 = False


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
        y_hat = numpy.array([1,2,3])
        y_obs = numpy.array([4,5,6])
        errormodel = calibr8.ErrorModel(independent, dependent)
        with self.assertRaises(NotImplementedError):
            _ = errormodel.predict_dependent(y_hat)
        with self.assertRaises(NotImplementedError):
            _ = errormodel.predict_independent(y_hat)
        with self.assertRaises(NotImplementedError):
            _ = errormodel.infer_independent(y_obs)
        with self.assertRaises(NotImplementedError):
            _ = errormodel.loglikelihood(y_obs=y_obs, y_hat=y_hat, theta=[1,2,3])
        with self.assertRaises(NotImplementedError):
            _ = errormodel.fit(independent=y_hat, dependent=y_obs, theta_guessed=None)
        return
    

class LogisticTest(unittest.TestCase):
    def test_logistic(self):
        y_hat = numpy.array([1.,2.,4.])
        theta = [2,2,4,1]
        expected = 2*2-4+(2*(4-2))/(1+numpy.exp(-2*1/(4-2)*(y_hat-2)))
        true = calibr8.logistic(y_hat, theta)
        self.assertTrue(numpy.array_equal(true, expected))
        return

    def test_inverse_logistic(self):
        y_hat = numpy.array([1.,2.,4.])
        theta = [2,2,4,1]
        forward = calibr8.logistic(y_hat, theta)
        reverse = calibr8.inverse_logistic(forward, theta)
        self.assertTrue(numpy.allclose(numpy.array(y_hat), reverse))
        return       

    def test_log_log_logistic(self):
        y_hat = numpy.array([1.,2.,4.])
        theta = [2,2,4,1]
        expected = numpy.exp(2*2-4+(2*(4-2))/(1+numpy.exp(-2*1/(4-2)*(numpy.log(y_hat)-2))))
        true = calibr8.log_log_logistic(y_hat, theta)
        self.assertTrue(numpy.array_equal(true, expected))
        expected = numpy.exp(calibr8.logistic(numpy.log(y_hat), theta))   
        self.assertTrue(numpy.array_equal(true, expected))
        return

    def test_inverse_log_log_logistic(self):
        y_hat = numpy.array([1.,2.,4.])
        theta = [2,2,4,1]
        forward = calibr8.log_log_logistic(y_hat, theta)
        reverse = calibr8.inverse_log_log_logistic(forward, theta)
        self.assertTrue(numpy.allclose(numpy.array(y_hat), reverse))
        return

    def test_xlog_logistic(self):
        y_hat = numpy.array([1.,2.,4.])
        theta = [2,2,4,1]
        expected = 2*2-4+(2*(4-2))/(1+numpy.exp(-2*1/(4-2)*(numpy.log(y_hat)-2)))
        true = calibr8.xlog_logistic(y_hat, theta)
        self.assertTrue(numpy.array_equal(true, expected))
        expected = calibr8.logistic(numpy.log(y_hat), theta)
        self.assertTrue(numpy.array_equal(true, expected))        
        return
        
    def test_inverse_xlog_logistic(self):
        y_hat = numpy.array([1.,2.,4.])
        theta = [2,2,4,1]
        forward = calibr8.xlog_logistic(y_hat, theta)
        reverse = calibr8.inverse_xlog_logistic(forward, theta)
        self.assertTrue(numpy.allclose(numpy.array(y_hat), reverse))
        return

    def test_ylog_logistic(self):
        y_hat = numpy.array([1.,2.,4.])
        theta = [2,2,4,1]
        expected = numpy.exp(2*2-4+(2*(4-2))/(1+numpy.exp(-2*1/(4-2)*(y_hat-2))))
        true = calibr8.ylog_logistic(y_hat, theta)
        self.assertTrue(numpy.array_equal(true, expected))
        expected = numpy.exp(calibr8.logistic(y_hat, theta))
        self.assertTrue(numpy.array_equal(true, expected))
        return

    def test_inverse_xlog_logistic(self):
        y_hat = numpy.array([1.,2.,4.])
        theta = [2,2,4,1]
        forward = calibr8.ylog_logistic(y_hat, theta)
        reverse = calibr8.inverse_ylog_logistic(forward, theta)
        self.assertTrue(numpy.allclose(numpy.array(y_hat), reverse))
        return


class GlucoseErrorModelTest(unittest.TestCase):
    def test_predict_dependent(self):
        independent = 'S'
        dependent = 'OD'
        y_hat = numpy.array([1,2,3])
        theta = [0,0,0]
        errormodel = calibr8.GlucoseErrorModel(independent, dependent)
        errormodel.theta_fitted = [0,1,0.1]
        with self.assertRaises(TypeError):
            _ = errormodel.predict_dependent(y_hat, theta)
        mu, sigma, df = errormodel.predict_dependent(y_hat)
        self.assertTrue(numpy.array_equal(mu, numpy.array([1,2,3])))
        self.assertTrue(numpy.array_equal(sigma, numpy.array([0.1,0.1,0.1])))
        self.assertEqual(df, 1)
        return
    
    def test_predict_independent(self):
        errormodel = calibr8.GlucoseErrorModel('S', 'OD')
        errormodel.theta_fitted = [0, 2, 0.1]
        
        x_original = numpy.array([4, 5, 6])
        mu, sd, df = errormodel.predict_dependent(x_original)
        x_predicted = errormodel.predict_independent(y_obs=mu)
        
        self.assertTrue(numpy.array_equal(mu, [8, 10, 12]))
        self.assertTrue(numpy.array_equal(sd, [0.1, 0.1, 0.1]))
        self.assertTrue(numpy.allclose(x_predicted, x_original))
        return
    
    @unittest.skipUnless(HAVE_PYMC3, "requires PyMC3")
    def test_infer_independent(self):
        errormodel = calibr8.GlucoseErrorModel('S', 'OD')
        errormodel.theta_fitted = [0, 2, 0.1]
        trace = errormodel.infer_independent(y_obs=1, draws=1)
        self.assertTrue(len(trace)==1)
        self.assertTrue(len(trace['Glucose'][0]==1))
        return

    @unittest.skipIf(HAVE_PYMC3, "only if PyMC3 is not imported")
    def test_error_infer_independent(self):
        errormodel = calibr8.GlucoseErrorModel('S', 'OD')
        with self.assertRaises(ImportError):
            _ = errormodel.infer_independent(y_obs=1, draws=1)
        return

    def test_loglikelihood(self):
        independent = 'S'
        dependent = 'OD'
        y_hat = numpy.array([1,2,3])
        y_obs = numpy.array([1,2,3])
        errormodel = calibr8.GlucoseErrorModel(independent, dependent)
        errormodel.theta_fitted = [0,1,0.1]
        with self.assertRaises(TypeError):
            _ = errormodel.loglikelihood(y_obs, y_hat=y_hat)
        true = errormodel.loglikelihood(y_obs=y_obs, y_hat=y_hat)
        mu, sigma, df = errormodel.predict_dependent(y_hat, theta=errormodel.theta_fitted)
        expected = numpy.sum(numpy.log(stats.t.pdf(x=y_obs, loc=mu, scale=sigma, df=1)))
        self.assertEqual(expected, true)
        true = errormodel.loglikelihood(y_obs=y_obs, y_hat=y_hat)
        mu, sigma, df = errormodel.predict_dependent(y_hat, theta=errormodel.theta_fitted)
        expected = numpy.sum(numpy.log(stats.t.pdf(x=y_obs, loc=mu, scale=sigma, df=1)))
        self.assertEqual(expected, true)
        return
    
    def test_loglikelihood_without_fit(self):
        independent = 'Glu'
        dependent = 'OD'
        y_hat = numpy.array([1,2,3])
        y_obs = numpy.array([1,2,3])
        errormodel = calibr8.GlucoseErrorModel(independent, dependent)
        with self.assertRaises(Exception):
            _= errormodel.loglikelihood(y_obs=y_obs, y_hat=y_hat)
        return


class BiomassErrorModelTest(unittest.TestCase):
    def test_predict_dependent(self):
        independent = 'BTM'
        dependent = 'BS'
        y_hat = numpy.array([1,10])
        errormodel = calibr8.BiomassErrorModel(independent, dependent)
        errormodel.theta_fitted = numpy.array([5,5,10,0.5,1,0])
        theta = numpy.array([0,0,0,0,0])
        with self.assertRaises(TypeError):
            _ = errormodel.predict_dependent(y_hat, theta)
        mu, sigma, df = errormodel.predict_dependent(y_hat)
        expected = numpy.exp(2*5-10+(2*(10-5))/(1+numpy.exp(-2*0.5/(10-5)*(numpy.log(y_hat)-5))))
        self.assertTrue(numpy.allclose(mu, expected))
        self.assertTrue(numpy.allclose(sigma, numpy.array([1,1])))
        self.assertEqual(df, 1)
        return
    
    def test_predict_independent(self):
        errormodel = calibr8.BiomassErrorModel('X', 'BS')
        errormodel.theta_fitted = numpy.array([5, 5, 10, 0.5, 0, 1])
        
        x_original = numpy.linspace(0.01, 30, 20)
        mu, sd, df = errormodel.predict_dependent(x_original)
        x_predicted = errormodel.predict_independent(y_obs=mu)
        
        self.assertTrue(numpy.allclose(x_predicted, x_original))
        return

    @unittest.skipUnless(HAVE_PYMC3, "requires PyMC3")
    def test_infer_independent(self):
        errormodel = calibr8.BiomassErrorModel('X', 'BS')
        errormodel.theta_fitted = numpy.array([5, 5, 10, 0.5, 0, 1])
        trace = errormodel.infer_independent(y_obs=1, draws=1)
        self.assertTrue(len(trace)==1)
        self.assertTrue(len(trace['BTM'][0]==1))
        return

    @unittest.skipIf(HAVE_PYMC3, "only if PyMC3 is not imported")
    def test_error_infer_independent(self):
        errormodel = calibr8.BiomassErrorModel('X', 'BS')
        with self.assertRaises(ImportError):
            errormodel.infer_independent(1)
        return

    def test_loglikelihood(self):
        independent = 'X'
        dependent = 'BS'
        y_hat = numpy.array([1,2,3])
        y_obs = numpy.array([1,2,3])
        errormodel = calibr8.BiomassErrorModel(independent, dependent)
        errormodel.theta_fitted = numpy.array([5,5,10,0.5,0,1])
        with self.assertRaises(TypeError):
            _ = errormodel.loglikelihood(y_obs, y_hat=y_hat, theta = errormodel.theta_fitted)
        theta = errormodel.theta_fitted
        true = errormodel.loglikelihood(y_obs=y_obs, y_hat=y_hat, theta=theta)
        mu, sigma, df = errormodel.predict_dependent(y_hat, theta=theta)
        expected = numpy.sum(numpy.log(stats.t.pdf(x=y_obs, loc=mu, scale=sigma, df=1)))
        self.assertEqual(expected, true)
        true = errormodel.loglikelihood(y_obs=y_obs, y_hat=y_hat)
        mu, sigma, df = errormodel.predict_dependent(y_hat, theta=theta)
        expected = numpy.sum(numpy.log(stats.t.pdf(x=y_obs, loc=mu, scale=sigma, df=1)))
        self.assertEqual(expected, true)
        return
    
    def test_loglikelihood_without_fit(self):
        independent = 'X'
        dependent = 'BS'
        y_hat = numpy.array([1,2,3])
        y_obs = numpy.array([1,2,3])
        errormodel = calibr8.BiomassErrorModel(independent, dependent)
        with self.assertRaises(Exception):
            _= errormodel.loglikelihood(y_obs=y_obs, y_hat=y_hat)
        return


if __name__ == '__main__':
    unittest.main(exit=False)