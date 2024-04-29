import unittest
from automl import MLSystem

class TestMLSystem(unittest.TestCase):
    def setUp(self):
        self.ml_system = MLSystem()
        self.train_data_path = "data/train.csv"
        self.test_data_path = "data/test.csv"

    def test_load_data(self):
        data = self.ml_system.load_data(self.train_data_path)
        self.assertIsNotNone(data)
        self.assertTrue(isinstance(data, pd.DataFrame))

    def test_preprocess_data(self):
        data = self.ml_system.load_data(self.train_data_path)
        processed_data = self.ml_system.preprocess_data(data)
        self.assertIsNotNone(processed_data)
        self.assertTrue(isinstance(processed_data, dict))

    def test_tuned_model(self):
        data = self.ml_system.load_data(self.train_data_path)
        processed_data = self.ml_system.preprocess_data(data)
        tuned_model = self.ml_system.tuned_model(processed_data)
        self.assertIsNotNone(tuned_model)

    def test_evaluate_model(self):
        data = self.ml_system.load_data(self.train_data_path)
        processed_data = self.ml_system.preprocess_data(data)
        tuned_model = self.ml_system.tuned_model(processed_data)
        _, rmsle = self.ml_system.evaluate_model(tuned_model, self.test_data_path)
        self.assertIsNotNone(rmsle)
        self.assertTrue(isinstance(rmsle, float))

    def test_run_entire_workflow(self):
        result = self.ml_system.run_entire_workflow(self.train_data_path, self.test_data_path)
        self.assertIn('RMSLE', result)
        self.assertTrue(isinstance(result['RMSLE'], float))

if __name__ == '__main__':
    unittest.main()
