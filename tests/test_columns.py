"""Analytic controls for Gaussian source and fixed detector integration."""
import unittest
from observation.columns.checks import run as pencil_checks
from observation.columns.adaptive_checks import run as detector_checks


class ColumnTests(unittest.TestCase):
    def test_pencil_source_analytic_controls(self):
        pencil_checks()

    def test_detector_area_and_source_failure_controls(self):
        detector_checks()
