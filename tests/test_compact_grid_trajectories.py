import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.compact_grid_trajectories import compact_group, load_trajectory


class CompactGridTrajectoriesTests(unittest.TestCase):
    def test_archive_read_and_replace_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            revolution = Path(directory) / "dep_2028-01-01" / "N0"
            run = revolution / "tof_500" / "cold"
            trajectory = run / "pythagorean_sundman" / "trajectory.csv"
            trajectory.parent.mkdir(parents=True)
            solution = trajectory.with_name("solution.json")
            solution.write_text('{"status":"Solve_Succeeded"}')
            first = np.arange(30.0).reshape(2, 15)
            np.savetxt(trajectory, first, delimiter=",", header="columns", comments="")

            count, original, archived = compact_group(revolution)
            self.assertEqual(count, 1)
            self.assertLess(archived, original)
            self.assertFalse(trajectory.exists())
            self.assertTrue(solution.exists())
            np.testing.assert_array_equal(load_trajectory(run), first)

            second = first + 1
            np.savetxt(trajectory, second, delimiter=",", header="columns", comments="")
            self.assertEqual(compact_group(revolution)[0], 1)
            np.testing.assert_array_equal(load_trajectory(run), second)

    def test_porkchop_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            departure = Path(directory) / "dep_2028-01-01"
            run = departure / "tof_500" / "N0"
            trajectory = run / "pythagorean_sundman" / "trajectory.csv"
            trajectory.parent.mkdir(parents=True)
            expected = np.arange(30.0).reshape(2, 15)
            np.savetxt(trajectory, expected, delimiter=",", header="columns", comments="")
            self.assertEqual(compact_group(departure)[0], 1)
            self.assertFalse(trajectory.exists())
            np.testing.assert_array_equal(load_trajectory(run), expected)


if __name__ == "__main__":
    unittest.main()
