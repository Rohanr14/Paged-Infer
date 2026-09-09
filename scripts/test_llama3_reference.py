"""The oracle checker must reject metadata drift and corrupted numeric outputs."""

import copy
import json
import unittest

from gen_llama3_reference import SMALL_OUTPUT, check_manifest, check_rope


class ReferenceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((SMALL_OUTPUT / "manifest.json").read_text())
        cls.rope = json.loads((SMALL_OUTPUT / "rope.json").read_text())

    def test_original_fixtures_are_accepted(self):
        check_manifest(self.manifest, copy.deepcopy(self.manifest))
        check_rope(self.rope, copy.deepcopy(self.rope))

    def test_config_floats_are_exact_not_numeric_outputs(self):
        changed = copy.deepcopy(self.manifest)
        changed["config"]["rms_norm_eps"] += 1e-8
        with self.assertRaises(ValueError):
            check_manifest(self.manifest, changed)
        changed = copy.deepcopy(self.rope)
        changed["cases"][0]["config"]["rope_scaling"]["factor"] += 1e-7
        with self.assertRaises(ValueError):
            check_rope(self.rope, changed)

    def test_checkpoint_identity_cannot_drift(self):
        changed = copy.deepcopy(self.manifest)
        changed["checkpoint_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            check_manifest(self.manifest, changed)

    def test_missing_frequency_cannot_hide_behind_absolute_epsilon(self):
        changed = copy.deepcopy(self.rope)
        changed["cases"][0]["inv_freq"][-1] = 0.0
        with self.assertRaises(ValueError):
            check_rope(self.rope, changed)

    def test_corrupt_table_and_rotation_are_rejected(self):
        for key in ["cos", "sin", "q_rotated", "k_rotated"]:
            with self.subTest(key=key):
                changed = copy.deepcopy(self.rope)
                changed["cases"][0]["positions"][0][key][0] += 0.001
                with self.assertRaises(ValueError):
                    check_rope(self.rope, changed)

    def test_missing_case_and_nonfinite_outputs_are_rejected(self):
        changed = copy.deepcopy(self.rope)
        changed["cases"].pop()
        with self.assertRaises(ValueError):
            check_rope(self.rope, changed)
        changed = copy.deepcopy(self.rope)
        changed["cases"][0]["positions"][0]["cos"][0] = float("nan")
        with self.assertRaises(ValueError):
            check_rope(self.rope, changed)


if __name__ == "__main__":
    unittest.main()
