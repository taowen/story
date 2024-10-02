import unittest
import get_hash_key

def my_func1():
    pass

def my_func2():
    pass

class TestFlow(unittest.TestCase):

    def test_get_hash_key(self):
        # different function name
        self.assertNotEqual(get_hash_key.get_hash_key(my_func1, [], {})[0], get_hash_key.get_hash_key(my_func2, [], {})[0])
