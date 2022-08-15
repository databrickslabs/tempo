from tempo import TSDF
from tempo.resample import checkAllowableFreq, _appendAggKey
from tests.base import SparkTest


class ResampleUnitTests(SparkTest):
    def test_appendAggKey_freq_is_none(self):
        input_tsdf = self.get_data_as_tsdf("input_data")

        self.assertRaises(TypeError, _appendAggKey, input_tsdf)

    def test_appendAggKey_freq(self):
        input_tsdf = self.get_data_as_tsdf("input_data")

        appendAggKey_tsdf = _appendAggKey(input_tsdf, "1 MICROSECOND")

        self.assertIsInstance(appendAggKey_tsdf[0], TSDF)
        self.assertEqual(appendAggKey_tsdf[1], "1")
        self.assertEqual(appendAggKey_tsdf[2], "microseconds")

    def test_aggregate(self):
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
