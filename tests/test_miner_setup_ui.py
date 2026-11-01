"""zils-miner setup's base-profile selector and doctor's message on a machine that can't mine."""

import contextlib
import io
import unittest
from unittest.mock import patch

from zils import miner

OPTIONS = [("a", "first"), ("b", "second"), ("c", "third")]


class MenuKeyTests(unittest.TestCase):
    def test_arrows_and_vim_keys_move_and_wrap(self):
        self.assertEqual(miner.menu_key(0, "down", 3), (1, False))
        self.assertEqual(miner.menu_key(1, "j", 3), (2, False))
        self.assertEqual(miner.menu_key(2, "down", 3), (0, False))
        self.assertEqual(miner.menu_key(0, "up", 3), (2, False))
        self.assertEqual(miner.menu_key(2, "k", 3), (1, False))

    def test_a_number_jumps_and_enter_chooses(self):
        self.assertEqual(miner.menu_key(0, "3", 3), (2, False))
        self.assertEqual(miner.menu_key(0, "9", 3), (0, False))
        self.assertEqual(miner.menu_key(1, "enter", 3), (1, True))
        self.assertEqual(miner.menu_key(1, "x", 3), (1, False))

    def test_arrows_and_numbers_skip_closed_options(self):
        self.assertEqual(miner.menu_key(0, "down", 3, closed={1}), (2, False))
        self.assertEqual(miner.menu_key(2, "down", 4, closed={3}), (0, False))
        self.assertEqual(miner.menu_key(0, "up", 4, closed={2, 3}), (1, False))
        self.assertEqual(miner.menu_key(0, "2", 3, closed={1}), (0, False))

    def test_ctrl_c_stops(self):
        with self.assertRaises(KeyboardInterrupt):
            miner.menu_key(0, "ctrl-c", 3)


class ChooseWithoutATerminalTests(unittest.TestCase):
    def choose(self, *answers, default="b"):
        replies, out = iter(answers), io.StringIO()
        with patch("builtins.input", lambda prompt: next(replies)), contextlib.redirect_stdout(out):
            value = miner.choose("Base profile", OPTIONS, default)
        return value, out.getvalue()

    def test_lists_numbered_options_and_enter_keeps_the_default(self):
        value, out = self.choose("")
        self.assertEqual(value, "b")
        self.assertIn("  1. first", out)
        self.assertIn("  3. third", out)

    def test_a_number_or_a_value_picks_and_anything_else_asks_again(self):
        self.assertEqual(self.choose("3")[0], "c")
        self.assertEqual(self.choose("a")[0], "a")
        value, out = self.choose("nope", "4", "2")
        self.assertEqual(value, "b")
        self.assertEqual(out.count("pick 1 to 3"), 2)


class ClosedOptionsWithoutATerminalTests(unittest.TestCase):
    def test_a_closed_option_is_listed_but_not_chosen(self):
        replies, out = iter(["3", "c", "1"]), io.StringIO()
        with patch("builtins.input", lambda prompt: next(replies)), contextlib.redirect_stdout(out):
            value = miner.choose("Base profile", OPTIONS, "c", closed={"c"})
        self.assertEqual(value, "a")
        self.assertIn("  3. third (opens later)", out.getvalue())
        self.assertEqual(out.getvalue().count("c opens later"), 2)

    def test_a_closed_default_falls_back_to_the_first_open_option(self):
        replies, out = iter([""]), io.StringIO()
        with patch("builtins.input", lambda prompt: next(replies)), contextlib.redirect_stdout(out):
            value = miner.choose("Base profile", OPTIONS, "a", closed={"a"})
        self.assertEqual(value, "b")


class SetupProfilesTests(unittest.TestCase):
    def test_the_menu_lists_the_launch_pools_with_unopened_ones_closed(self):
        profiles, closed = miner.setup_profiles()
        values = [value for value, _ in profiles]
        self.assertEqual(
            values, ["d3-lite-0.8b-v1", "d3-mini-4.5b-v1", "d3-flash-8.4b-v1", "d3-ultra-26b-v1"]
        )
        self.assertNotIn("d3-nano-2.2b-v1", values)
        self.assertEqual(closed, {"d3-flash-8.4b-v1", "d3-ultra-26b-v1"})


class MachineThatCannotMineTests(unittest.TestCase):
    def message(self, system, machine):
        with (
            patch("zils.miner.platform.system", return_value=system),
            patch("zils.miner.platform.machine", return_value=machine),
        ):
            return miner.missing_runtime("/home/x/.zils/zils/.venv-d3/bin/python")

    def test_a_mac_is_told_plainly_to_use_a_gpu_machine(self):
        text = self.message("Darwin", "arm64")
        self.assertIn("This Mac can't mine", text)
        self.assertIn("Linux x86_64 with an NVIDIA GPU", text)
        self.assertIn("on that machine", text)
        self.assertIn(miner.INSTALL, text)
        self.assertNotIn(".venv-d3", text)

    def test_other_unsupported_machines_are_named(self):
        self.assertIn("This Windows machine can't mine", self.message("Windows", "AMD64"))
        self.assertIn("This Linux aarch64 machine can't mine", self.message("Linux", "aarch64"))

    def test_a_linux_gpu_host_is_told_to_rerun_the_installer_there(self):
        text = self.message("Linux", "x86_64")
        self.assertIn("no d3 training runtime yet at /home/x/.zils/zils/.venv-d3/bin/python", text)
        self.assertIn("rerun the installer on this machine", text)
        self.assertIn(miner.INSTALL, text)


if __name__ == "__main__":
    unittest.main()
