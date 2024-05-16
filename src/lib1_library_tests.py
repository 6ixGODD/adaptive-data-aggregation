import unittest  # The test framework

import lab1_library  # The code to test


# Example of quantization function tests
class Test_TestQuantizationOfErrors(unittest.TestCase):
    def test_quantize_returns_zero_for_1bit(self):
        self.assertEqual(lab1_library.quantize(0, 1), 0)

    def test_quantize_returns_zero_for_10bit(self):
        self.assertEqual(lab1_library.quantize(0, 10), 0)

    def test_quantize_rounds_to_integer_for_10bit(self):
        self.assertEqual(lab1_library.quantize(11.4, 10), 11)

    def test_quantize_saturates_at_minimum_for_8bit(self):
        self.assertEqual(lab1_library.quantize(-129, 8), -128)

    def test_quantize_saturates_at_maximum_for_9bit(self):
        self.assertEqual(lab1_library.quantize(40000, 9), +255)


if __name__ == '__main__':
    unittest.main()
