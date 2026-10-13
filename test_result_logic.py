import unittest

from result_logic import FAIL_RESULT, PASS_RESULT, predict_result


class TestResultLogic(unittest.TestCase):

    def test_pass_case(self):
        result = predict_result(70, 85)
        self.assertEqual(result, PASS_RESULT)

    def test_fail_due_to_marks(self):
        result = predict_result(30, 85)
        self.assertEqual(result, FAIL_RESULT)

    def test_fail_due_to_attendance(self):
        result = predict_result(70, 60)
        self.assertEqual(result, FAIL_RESULT)

    def test_result_constants_are_expected_values(self):
        self.assertEqual(PASS_RESULT, "PASS")
        self.assertEqual(FAIL_RESULT, "FAIL")


if __name__ == "__main__":
    unittest.main()