"""Capabilities each pinned profile advertises to the platform."""

import subprocess
import sys
import unittest

from zils import models
from zils.image_contract import IMAGE_CAPABILITIES, TEXT_CAPABILITIES, capabilities


class CapabilitiesTest(unittest.TestCase):
    def test_capabilities_per_profile(self):
        self.assertIs(capabilities(models.IMAJEV), IMAGE_CAPABILITIES)
        self.assertEqual(
            capabilities(models.D3),
            {
                "modalities": ["text", "image", "video"],
                "question_types": ["noul", "choice", "score"],
                "max_images": 4,
                "max_videos": 1,
                "max_questions": 8,
                "max_options": 255,
                "max_input_tokens": 4096,
                "unknown_key": None,
            },
        )
        for model in (models.H2O,):
            self.assertIs(capabilities(model), TEXT_CAPABILITIES)

    def test_import_stays_free_of_torch_and_pil(self):
        code = (
            "import sys, zils.image_contract; print('torch' in sys.modules, 'PIL' in sys.modules)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        self.assertEqual(out.stdout.split(), ["False", "False"])


if __name__ == "__main__":
    unittest.main()
