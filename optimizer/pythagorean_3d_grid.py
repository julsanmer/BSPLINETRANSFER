"""Node-discretized alternative to the 3D Pythagorean Bernstein solver.

This entry point uses the same trajectory variables, SPICE transformation,
boundary conditions, and minimum-peak objective as
``optimizer.pythagorean_3d_bernstein``.  Only thrust enforcement changes:
the acceleration norm is bounded at equally spaced nodes on each spline span.
Dense sampling and an a-posteriori continuous Bernstein bound are reported to
quantify missed inter-node peaks and certificate conservatism.
"""

from optimizer.pythagorean_3d_bernstein import main


if __name__ == "__main__":
    main(default_thrust_enforcement="grid")
