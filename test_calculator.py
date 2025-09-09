"""
Comprehensive test suite for the calculator module.
This demonstrates various Python testing patterns and best practices.
"""

import unittest
from unittest.mock import patch, MagicMock
from calculator import Calculator, factorial

# Import pytest only if available
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Create a dummy pytest module for the examples
    class DummyPytest:
        class mark:
            @staticmethod
            def parametrize(*args, **kwargs):
                def decorator(func):
                    return func
                return decorator
        
        @staticmethod
        def raises(*args, **kwargs):
            class DummyContext:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            return DummyContext()
    
    pytest = DummyPytest()


class TestCalculator(unittest.TestCase):
    """Test cases for the Calculator class using unittest framework."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.calc = Calculator()
    
    def tearDown(self):
        """Clean up after each test method."""
        # In this simple case, no cleanup is needed
        pass
    
    def test_add_positive_numbers(self):
        """Test addition of positive numbers."""
        result = self.calc.add(2, 3)
        self.assertEqual(result, 5)
    
    def test_add_negative_numbers(self):
        """Test addition with negative numbers."""
        result = self.calc.add(-2, -3)
        self.assertEqual(result, -5)
    
    def test_add_mixed_numbers(self):
        """Test addition of positive and negative numbers."""
        result = self.calc.add(-2, 3)
        self.assertEqual(result, 1)
    
    def test_subtract(self):
        """Test subtraction operation."""
        self.assertEqual(self.calc.subtract(5, 3), 2)
        self.assertEqual(self.calc.subtract(3, 5), -2)
        self.assertEqual(self.calc.subtract(-2, -3), 1)
    
    def test_multiply(self):
        """Test multiplication operation."""
        self.assertEqual(self.calc.multiply(3, 4), 12)
        self.assertEqual(self.calc.multiply(-2, 3), -6)
        self.assertEqual(self.calc.multiply(0, 5), 0)
    
    def test_divide_normal_cases(self):
        """Test normal division cases."""
        self.assertEqual(self.calc.divide(10, 2), 5.0)
        self.assertEqual(self.calc.divide(7, 2), 3.5)
        self.assertEqual(self.calc.divide(-8, 4), -2.0)
    
    def test_divide_by_zero(self):
        """Test that division by zero raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.calc.divide(10, 0)
        self.assertEqual(str(context.exception), "Cannot divide by zero")
    
    def test_power(self):
        """Test power operation."""
        self.assertEqual(self.calc.power(2, 3), 8)
        self.assertEqual(self.calc.power(5, 0), 1)
        self.assertEqual(self.calc.power(3, 2), 9)
    
    def test_is_even(self):
        """Test even number detection."""
        self.assertTrue(self.calc.is_even(4))
        self.assertTrue(self.calc.is_even(0))
        self.assertTrue(self.calc.is_even(-2))
        self.assertFalse(self.calc.is_even(3))
        self.assertFalse(self.calc.is_even(-1))


class TestFactorial(unittest.TestCase):
    """Test cases for the factorial function."""
    
    def test_factorial_positive_numbers(self):
        """Test factorial for positive numbers."""
        self.assertEqual(factorial(0), 1)
        self.assertEqual(factorial(1), 1)
        self.assertEqual(factorial(5), 120)
        self.assertEqual(factorial(3), 6)
    
    def test_factorial_negative_number(self):
        """Test that factorial of negative number raises ValueError."""
        with self.assertRaises(ValueError) as context:
            factorial(-1)
        self.assertEqual(str(context.exception), "Factorial is not defined for negative numbers")


# Pytest-style tests (can be run with pytest)
class TestCalculatorPytest:
    """Test cases using pytest framework."""
    
    def setup_method(self):
        """Setup method called before each test."""
        self.calc = Calculator()
    
    def test_add_with_fixtures(self):
        """Test addition using pytest style."""
        assert self.calc.add(2, 3) == 5
        assert self.calc.add(-1, 1) == 0
    
    @pytest.mark.parametrize("a,b,expected", [
        (2, 3, 5),
        (-1, 1, 0),
        (0, 0, 0),
        (10, -5, 5),
    ])
    def test_add_parametrized(self, a, b, expected):
        """Test addition with multiple parameter sets."""
        assert self.calc.add(a, b) == expected
    
    @pytest.mark.parametrize("base,exp,expected", [
        (2, 3, 8),
        (5, 0, 1),
        (1, 100, 1),
        (3, 2, 9),
    ])
    def test_power_parametrized(self, base, exp, expected):
        """Test power operation with multiple parameter sets."""
        assert self.calc.power(base, exp) == expected
    
    def test_divide_by_zero_pytest(self):
        """Test division by zero using pytest."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            self.calc.divide(10, 0)


# Mock testing example
class TestCalculatorWithMocks(unittest.TestCase):
    """Demonstrate testing with mocks."""
    
    def setUp(self):
        self.calc = Calculator()
    
    @patch('calculator.Calculator.add')
    def test_mocked_add(self, mock_add):
        """Test using a mocked method."""
        # Configure the mock
        mock_add.return_value = 100
        
        # Call the method
        result = self.calc.add(2, 3)
        
        # Verify the mock was called and returned expected value
        mock_add.assert_called_once_with(2, 3)
        self.assertEqual(result, 100)


# Test runner functions
def run_unittest_tests():
    """Run tests using unittest framework."""
    print("Running unittest tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)


def run_pytest_tests():
    """Run tests using pytest framework."""
    if not PYTEST_AVAILABLE:
        print("Pytest is not available. Install it with: pip install pytest")
        return
        
    print("Running pytest tests...")
    import subprocess
    import sys
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            __file__, 
            '-v',  # verbose output
            '--tb=short'  # short traceback format
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
    except Exception as e:
        print(f"Error running pytest: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("PYTHON TEST CODE DEMONSTRATION")
    print("=" * 60)
    
    # Show different ways to run tests
    print("\n1. Running with unittest framework:")
    print("-" * 40)
    
    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestFactorial))
    suite.addTests(loader.loadTestsFromTestCase(TestCalculatorWithMocks))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    print("\n" + "=" * 60)
    print("To run these tests, you can use:")
    print("1. python test_calculator.py")
    print("2. python -m unittest test_calculator.py")
    print("3. pytest test_calculator.py (requires: pip install pytest)")
    print("=" * 60)