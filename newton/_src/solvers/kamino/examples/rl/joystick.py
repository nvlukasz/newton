# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""General-purpose Xbox gamepad / keyboard controller for RL inference.

Reads Xbox 360/One controller input (or keyboard via the viewer) and
provides velocity commands and head pose deltas.  Optionally integrates
a 2-D path (heading + position) from the velocity commands.

Input is selected automatically:

1. Xbox gamepad (if ``xbox360controller`` package is installed and a pad is
   connected).
2. Keyboard via the 3D viewer window (if a viewer is provided).
3. No-op — zero commands, robot stands still.

Gamepad mapping::

    Left stick Y          forward / backward velocity
    Left stick X          yaw rate (angular velocity)
    Triggers (L minus R)  lateral velocity
    Right stick Y         head pitch (look up / down)
    Right stick X         head yaw (look left / right)
    RB (right bumper)     turbo (hold for higher velocity limits)
    Select / Back         reset

Keyboard layout (viewer window must have focus)::

    I / K  — forward / backward
    J / L  — strafe left / right
    U / O  — turn left / right
    T / G  — head pitch up / down
    F / H  — head yaw left / right
    P      — reset
"""

from __future__ import annotations

# Python
import dataclasses

# Thirdparty
import torch  # noqa: TID253
import warp as wp

from newton._src.solvers.kamino.examples.rl.utils import RateLimitedValue, _deadband, _LowPassFilter, _scale_asym

# ---------------------------------------------------------------------------
# Warp kernel: mid-point path integration + deviation clipping (all worlds)
# ---------------------------------------------------------------------------
_Z_AXIS = wp.constant(wp.vec3(0.0, 0.0, 1.0))


@wp.kernel
def _integrate_path(
    path_heading: wp.array(dtype=wp.float32),   # (num_worlds,)      flat, updated in-place
    path_position: wp.array(dtype=wp.float32),  # (num_worlds * 2,)  flat, updated in-place
    root_pos_2d: wp.array(dtype=wp.float32),    # (num_worlds * 2,)  flat, read-only
    fwd_vel: float,
    lat_vel: float,
    ang_vel: float,
    dt: float,
    path_deviation_max: float,
):
    w = wp.tid()
    pp = w * 2

    # Mid-point heading for more accurate integration
    mid_h = path_heading[w] + float(0.5) * dt * ang_vel

    # Rotate body-frame velocity into world frame
    q = wp.quat_from_axis_angle(_Z_AXIS, mid_h)
    vel_w = wp.quat_rotate(q, wp.vec3(fwd_vel, lat_vel, float(0.0)))

    # Integrate position
    new_px = path_position[pp + 0] + vel_w[0] * dt
    new_py = path_position[pp + 1] + vel_w[1] * dt

    # Update heading
    path_heading[w] = path_heading[w] + ang_vel * dt

    # Clip XY deviation from the robot's current ground position
    rx = root_pos_2d[pp + 0]
    ry = root_pos_2d[pp + 1]
    dx = new_px - rx
    dy = new_py - ry
    dist = wp.sqrt(dx * dx + dy * dy)
    if dist > path_deviation_max:
        scale = path_deviation_max / dist
        dx = dx * scale
        dy = dy * scale
    path_position[pp + 0] = rx + dx
    path_position[pp + 1] = ry + dy


@dataclasses.dataclass
class JoystickConfig:
    """Velocity limits and turbo parameters for :class:`JoystickController`.

    Each velocity axis has a *base* value (no turbo) and a *turbo* delta
    that is blended in as turbo_alpha ramps from 0 → 1::

        effective_max = base + turbo_alpha * turbo
    """

    # Velocity limits
    forward_velocity_base: float = 0.3
    forward_velocity_turbo: float = 0.3
    lateral_velocity_base: float = 0.15
    lateral_velocity_turbo: float = 0.15
    angular_velocity_base: float = 1.0
    angular_velocity_turbo: float = 0.75

    # Head limits
    head_pitch_up: float = 1.0
    head_pitch_down: float = 0.6
    head_yaw_max: float = 0.9

    # Input processing
    axis_deadband: float = 0.2
    trigger_deadband: float = 0.2
    cutoff_hz: float = 10.0

    # Path integration
    path_deviation_max: float = 0.1

    # Turbo ramp rate
    turbo_rate: float = 2.0

    def forward_velocity_max(self, turbo_alpha: float) -> float:
        return self.forward_velocity_base + turbo_alpha * self.forward_velocity_turbo

    def lateral_velocity_max(self, turbo_alpha: float) -> float:
        return self.lateral_velocity_base + turbo_alpha * self.lateral_velocity_turbo

    def angular_velocity_max(self, turbo_alpha: float) -> float:
        return self.angular_velocity_base + turbo_alpha * self.angular_velocity_turbo


class JoystickController:
    """General-purpose Xbox gamepad / keyboard controller for RL inference.

    Reads gamepad axes or keyboard keys and exposes command outputs as
    attributes.  Optionally integrates a 2-D path (heading + position)
    from the velocity commands.

    Gamepad mapping:
      Left stick Y          -> forward velocity
      Left stick X          -> yaw rate (angular velocity)
      Triggers (L minus R)  -> lateral velocity
      Right stick Y         -> neck pitch
      Right stick X         -> neck yaw
      RB (right bumper)     -> turbo (hold)
      Select / Back         -> reset

    Keyboard mapping (see module docstring for layout).

    Output attributes (updated each :meth:`update` call):
      ``forward_velocity``  Forward velocity   (positive = forward)
      ``lateral_velocity``  Lateral velocity   (positive = strafe left)
      ``angular_velocity``  Angular velocity   (positive = turn left)
      ``head_pitch``        Head pitch command (positive = look up)
      ``head_yaw``          Head yaw command   (positive = look left)
      ``turbo_alpha``       Current turbo blend factor (0.0 - 1.0)

    Path state (when ``root_pos_2d`` is passed to :meth:`update`):
      ``path_heading``      Integrated heading  ``(num_worlds, 1)``
      ``path_position``     Integrated position ``(num_worlds, 2)``
    """

    def __init__(
        self,
        dt: float,
        viewer=None,
        num_worlds: int = 1,
        device: str = "cuda:0",
        config: JoystickConfig | None = None,
    ) -> None:
        cfg = config or JoystickConfig()
        self._cfg = cfg
        self._dt = dt
        self._viewer = viewer
        self._num_worlds = num_worlds
        self._device = device

        # Low-pass filters (named by semantic axis)
        hz = cfg.cutoff_hz
        self._forward_filter = _LowPassFilter(hz, dt)
        self._lateral_filter = _LowPassFilter(hz, dt)
        self._angular_filter = _LowPassFilter(hz, dt)
        self._head_pitch_filter = _LowPassFilter(hz, dt)
        self._head_yaw_filter = _LowPassFilter(hz, dt)

        # Turbo ramp (rate-limited 0→1 blend)
        self._turbo = RateLimitedValue(cfg.turbo_rate, dt)

        # Path state (per-world)
        self.path_heading = torch.zeros(num_worlds, 1, device=device)
        self.path_position = torch.zeros(num_worlds, 2, device=device)

        # Command outputs (updated by update())
        self.forward_velocity: float = 0.0
        self.lateral_velocity: float = 0.0
        self.angular_velocity: float = 0.0
        self.head_pitch: float = 0.0
        self.head_yaw: float = 0.0
        self.turbo_alpha: float = 0.0

        # Pre-cached Warp views for path state (no data copy, just pointer wrapping)
        self._wp_path_heading = wp.from_torch(self.path_heading.reshape(-1))
        self._wp_path_position = wp.from_torch(self.path_position.reshape(-1))

        # Use the same CUDA stream as PyTorch so the kernel is serialised with
        # subsequent torch/warp ops that read path_heading / path_position.
        self._wp_stream = (
            wp.stream_from_torch(torch.cuda.current_stream()) if "cuda" in str(device) else None
        )

        # Reset edge-detection state
        self._reset_prev = False
        # Camera button edge-detection state
        self._follow_toggle_prev = False
        self._snap_behind_prev = False
        self._snap_front_prev = False

        # --- Input mode detection ---
        self._controller = None
        self._mode: str | None = None  # "joystick", "keyboard", or None

        try:
            # Thirdparty
            from xbox360controller import Xbox360Controller  # noqa: PLC0415

            self._controller = Xbox360Controller(0, axis_threshold=0.015)
            self._mode = "joystick"
            print("Joystick connected.")
        except Exception:
            if viewer is not None and hasattr(viewer, "is_key_down"):
                self._mode = "keyboard"
                print(
                    "No joystick found. Using keyboard controls:\n"
                    "  I/K — forward/backward    J/L — strafe left/right\n"
                    "  U/O — turn left/right     T/G — look up/down\n"
                    "  F/H — look left/right\n"
                    "  P   — reset"
                )
            else:
                print("No joystick or keyboard available. Commands will be zero.")

    def _read_input(self) -> tuple[float, float, float, float, float]:
        """Read controller input as semantic axes.

        Returns:
            ``(forward, lateral, angular, head_pitch, head_yaw)``

        Sign convention — positive means:
          forward   : walk forward
          lateral   : strafe left
          angular   : turn left  (CCW)
          head_pitch: look up
          head_yaw  : look left
        """
        if self._mode == "joystick":
            c = self._controller
            return (
                -c.axis_l.y,  # forward   (negate: HW up is negative)
                c.trigger_l.value - c.trigger_r.value,  # lateral   (L trigger = strafe left)
                -c.axis_l.x,  # angular   (negate: HW left is negative)
                -c.axis_r.y,  # head pitch (negate: HW up is negative)
                -c.axis_r.x,  # head yaw   (negate: HW left is negative)
            )

        # Keyboard fallback
        v = self._viewer

        def _axis(neg_key: str, pos_key: str) -> float:
            val = 0.0
            if v.is_key_down(neg_key):
                val -= 1.0
            if v.is_key_down(pos_key):
                val += 1.0
            return val

        return (
            _axis("k", "i"),  # forward:    I = forward(+), K = backward(-)
            _axis("l", "j"),  # lateral:    J = left(+),    L = right(-)
            _axis("o", "u"),  # angular:    U = left(+),    O = right(-)
            _axis("t", "g"),  # head pitch: T = up(+),      G = down(-)
            _axis("h", "f"),  # head yaw:   F = left(+),    H = right(-)
        )

    def _read_turbo(self) -> float:
        """Return 1.0 if turbo is engaged, 0.0 otherwise."""
        if self._mode == "joystick":
            return 1.0 if self._controller.button_trigger_r.is_pressed else 0.0
        return 0.0

    def update(self, root_pos_2d: torch.Tensor | None = None) -> None:
        """Read input, compute commands, and optionally advance the path.

        Args:
            root_pos_2d: Current robot XY position ``(num_worlds, 2)`` for
                path deviation clipping.  When ``None``, path integration
                is skipped.
        """
        if self._mode is None:
            return

        cfg = self._cfg

        # --- Read & filter ---
        fwd_raw, lat_raw, ang_raw, npitch_raw, nyaw_raw = self._read_input()

        fwd = _deadband(self._forward_filter.update(fwd_raw), cfg.axis_deadband)
        lat = _deadband(self._lateral_filter.update(lat_raw), cfg.trigger_deadband)
        ang = _deadband(self._angular_filter.update(ang_raw), cfg.axis_deadband)
        npitch = _deadband(self._head_pitch_filter.update(npitch_raw), cfg.axis_deadband)
        nyaw = _deadband(self._head_yaw_filter.update(nyaw_raw), cfg.axis_deadband)

        # --- Turbo ---
        self.turbo_alpha = self._turbo.update(self._read_turbo())

        # --- Scale to physical units ---
        self.forward_velocity = fwd * cfg.forward_velocity_max(self.turbo_alpha)
        self.lateral_velocity = lat * cfg.lateral_velocity_max(self.turbo_alpha)
        self.angular_velocity = ang * cfg.angular_velocity_max(self.turbo_alpha)
        self.head_pitch = _scale_asym(npitch, cfg.head_pitch_down, cfg.head_pitch_up)
        self.head_yaw = nyaw * cfg.head_yaw_max

        # --- Path integration ---
        if root_pos_2d is not None:
            _wp_root = wp.from_torch(root_pos_2d.contiguous().reshape(-1))
            _launch_kw = (
                {"stream": self._wp_stream}
                if self._wp_stream is not None
                else {"device": str(self._device)}
            )
            wp.launch(
                _integrate_path,
                dim=self._num_worlds,
                inputs=[
                    self._wp_path_heading,
                    self._wp_path_position,
                    _wp_root,
                    self.forward_velocity,
                    self.lateral_velocity,
                    self.angular_velocity,
                    self._dt,
                    cfg.path_deviation_max,
                ],
                **_launch_kw,
            )

    def check_follow_toggle(self) -> bool:
        """Rising edge of follow-cam toggle: gamepad Y or keyboard ``x``."""
        pressed = False
        if self._mode == "joystick":
            pressed = bool(self._controller.button_y.is_pressed)
        if not pressed and self._viewer is not None and hasattr(self._viewer, "is_key_down"):
            pressed = bool(self._viewer.is_key_down("x"))
        triggered = pressed and not self._follow_toggle_prev
        self._follow_toggle_prev = pressed
        return triggered

    def check_snap_behind(self) -> bool:
        """Rising edge of snap-behind: gamepad A or keyboard ``shift``."""
        pressed = False
        if self._mode == "joystick":
            pressed = bool(self._controller.button_a.is_pressed)
        if not pressed and self._viewer is not None and hasattr(self._viewer, "is_key_down"):
            pressed = bool(self._viewer.is_key_down("shift"))
        triggered = pressed and not self._snap_behind_prev
        self._snap_behind_prev = pressed
        return triggered

    def check_snap_front(self) -> bool:
        """Rising edge of snap-to-front: gamepad B or keyboard ``ctrl``."""
        pressed = False
        if self._mode == "joystick":
            pressed = bool(self._controller.button_b.is_pressed)
        if not pressed and self._viewer is not None and hasattr(self._viewer, "is_key_down"):
            pressed = bool(self._viewer.is_key_down("ctrl"))
        triggered = pressed and not self._snap_front_prev
        self._snap_front_prev = pressed
        return triggered

    def close(self) -> None:
        """Release gamepad resources so the process can exit cleanly."""
        if self._controller is not None:
            try:
                self._controller.close()
            except Exception:
                pass
            self._controller = None

    def check_reset(self) -> bool:
        """Return True on the rising edge of the reset input.

        Gamepad: X button or Select/Back button.  Keyboard: ``p`` key.
        """
        pressed = False
        if self._mode == "joystick":
            c = self._controller
            pressed = bool(c.button_x.is_pressed) or bool(c.button_select.is_pressed)
            # Also allow keyboard 'p' when a gamepad is connected
            if not pressed and self._viewer is not None and hasattr(self._viewer, "is_key_down"):
                pressed = bool(self._viewer.is_key_down("p"))
        elif self._mode == "keyboard" and self._viewer is not None:
            pressed = bool(self._viewer.is_key_down("p"))
        triggered = pressed and not self._reset_prev
        self._reset_prev = pressed
        return triggered

    def reset(self, root_pos_2d: torch.Tensor | None = None, root_yaw: torch.Tensor | None = None) -> None:
        """Reset path state and filters.

        Args:
            root_pos_2d: Current robot XY position ``(num_worlds, 2)``.
            root_yaw: Current robot yaw angle ``(num_worlds, 1)``.
        """
        if root_yaw is not None:
            self.path_heading[:] = root_yaw
        if root_pos_2d is not None:
            self.path_position[:] = root_pos_2d

        self._forward_filter.reset()
        self._lateral_filter.reset()
        self._angular_filter.reset()
        self._head_pitch_filter.reset()
        self._head_yaw_filter.reset()
        self._turbo.reset()

    def set_dt(self, dt: float) -> None:
        """Change the timestep used for path integration and filtering."""
        self._dt = dt
        hz = self._cfg.cutoff_hz
        self._forward_filter = _LowPassFilter(hz, dt)
        self._lateral_filter = _LowPassFilter(hz, dt)
        self._angular_filter = _LowPassFilter(hz, dt)
        self._head_pitch_filter = _LowPassFilter(hz, dt)
        self._head_yaw_filter = _LowPassFilter(hz, dt)
        self._turbo.dt = dt
