"""Validate the restored input and corrected notebook structure.

These tests check data shape and syntax, not model performance or clinical validity.
"""
from __future__ import annotations

import ast
import csv
import json
import statistics
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DiabetesEDADataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "diabetes(1).csv").open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            cls.columns = reader.fieldnames
            cls.rows = list(reader)

    def test_shape_and_expected_columns(self):
        self.assertEqual(len(self.rows), 768)
        self.assertEqual(len(self.columns), 9)
        for key in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Age", "Outcome"]:
            self.assertIn(key, self.columns)

    def test_outcome_counts(self):
        self.assertEqual(sum(row["Outcome"] == "1" for row in self.rows), 268)
        self.assertEqual(sum(row["Outcome"] == "0" for row in self.rows), 500)

    def test_zero_counts_and_nonzero_median(self):
        insulin = [float(row["Insulin"]) for row in self.rows]
        self.assertEqual(sum(value == 0 for value in insulin), 374)
        self.assertEqual(statistics.median(insulin), 30.5)
        self.assertEqual(statistics.median(value for value in insulin if value != 0), 125)

    def test_corrected_notebook_all_code_cells_compile(self):
        notebook = json.loads((ROOT / "diabetes_visualisation_corrected.ipynb").read_text(encoding="utf-8"))
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                code = "".join(cell["source"])
                try:
                    ast.parse(code)
                except SyntaxError as exc:
                    self.fail(f"Code cell {i} has syntax error: {exc}")

    def test_final_findings_are_markdown_and_input_is_correct(self):
        notebook = json.loads((ROOT / "diabetes_visualisation_corrected.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["cells"][-1]["cell_type"], "markdown")
        code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
        self.assertIn('pd.read_csv("diabetes(1).csv")', code)
        self.assertIn('replace(0, np.nan)', code)


if __name__ == "__main__":
    unittest.main()
