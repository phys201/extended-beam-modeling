from unittest import TestCase
import xbmodeling

class TestSomethingGeneric(TestCase):
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

