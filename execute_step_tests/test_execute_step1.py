from execute_step import execute_step
import unittest

i = 9

def my_func1():
    global i
    i = i + 1
    return i


class TestFlow(unittest.TestCase):

    def test_execute_step(self):
        # should cache result
        result1 = execute_step(my_func1, (), {})
        result2 = execute_step(my_func1, (), {})
        self.assertEqual({"result": 10, "kwargs": {}}, result1)
        self.assertEqual({"result": 10, "kwargs": {}}, result2)
