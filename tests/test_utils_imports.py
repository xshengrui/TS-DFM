import subprocess
import sys
import unittest


class UtilsImportTests(unittest.TestCase):
    def test_alignment_import_does_not_require_optional_chemistry_packages(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from Utils.alignment import Kabsch_alignment; "
                "assert Kabsch_alignment.__name__ == 'Kabsch_alignment'",
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
