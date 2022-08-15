from tests.base import SparkTest
from tempo.resample import checkAllowableFreq


class ResampleUnitTests(SparkTest):
    def test__appendAggKey(self):
        pass

    def test_check_allowable_freq_none(self):
        self.assertRaises(TypeError, checkAllowableFreq, None)

    def test_check_allowable_freq_microsecond(self):
        self.assertEqual(checkAllowableFreq("1 MICROSECOND"), ("1", "microsec"))

    def test_check_allowable_freq_millisecond(self):
        self.assertEqual(checkAllowableFreq("1 MILLISECOND"), ("1", "ms"))

    def test_check_allowable_freq_second(self):
        self.assertEqual(checkAllowableFreq("1 SECOND"), ("1", "sec"))

    def test_check_allowable_freq_minute(self):
        self.assertEqual(checkAllowableFreq("1 MINUTE"), ("1", "min"))

    def test_check_allowable_freq_hour(self):
        self.assertEqual(checkAllowableFreq("1 HOUR"), ("1", "hour"))

    def test_check_allowable_freq_day(self):
        self.assertEqual(checkAllowableFreq("1 DAY"), ("1", "day"))

    def test_check_allowable_freq_no_interval(self):
        self.assertEqual(checkAllowableFreq("day"), (1, "day"))

    def test_check_allowable_freq_exception(self):
        self.assertRaises(ValueError, checkAllowableFreq, "wrong")
