import unittest

from tempo.utils import *

from tests.base import SparkTest


class UtilsTest(SparkTest):

    def test_display(self):
        """Test of the display utility"""
        if PLATFORM == 'DATABRICKS':
            self.assertEqual(id(display), id(display_improvised))
        elif ENV_BOOLEAN:
            self.assertEqual(id(display), id(display_html_improvised))
        else:
            self.assertEqual(id(display), id(display_unavailable))


# MAIN
if __name__ == '__main__':
    unittest.main()
