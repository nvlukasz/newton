# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scenes, runs, row read-back and kernel-text pins shared by the scene-mode test modules of SolverSteel."""

import re
from collections.abc import Sequence

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.steel._testing import kernel_overrides
from newton._src.solvers.steel.scene_rows import FAMILY_VELOCITY_LIMIT
from newton._src.solvers.steel.scene_solve import scene_internal_bearing
from newton._src.solvers.steel.scene_units import SceneUnits
from newton.solvers import SolverSteel
from newton.tests.steel._fixtures import CONTACT_MAX, DT, _build_plane_model


def _humanoid(*, lying=True) -> newton.ModelBuilder:
    """One ``nv_humanoid.xml`` humanoid (27 DOFs), the template of :func:`_humanoids`.

    ``lying`` turns its root 90 degrees about x at z = 0.375 m, onto its right side: after
    :func:`newton.eval_fk` its lowest collision shape, the right lower-arm capsule, is 4.0 mm
    above the ground, inside the builder's 5 mm ``rigid_gap``. Otherwise it stands with its root
    at z = 1.3 m. The builder registers Steel's custom attributes first.
    """
    humanoid = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverSteel.register_custom_attributes(humanoid)
    humanoid.rigid_gap = 0.005
    root = (
        wp.transform(wp.vec3(0.0, 0.0, 0.375), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * np.pi))
        if lying
        else wp.transform(wp.vec3(0.0, 0.0, 1.3), wp.quat_identity())
    )
    humanoid.add_mjcf(
        newton.examples.get_asset("nv_humanoid.xml"), xform=root, ignore_names=["floor", "ground"], up_axis="Z"
    )
    return humanoid


def _without_passive_springs(model: newton.Model) -> newton.Model:
    """Zero ``model``'s passive joint stiffness (:attr:`~newton.Model.joint_stiffness`); return it.

    The MJCF importer fills ``joint_stiffness`` from ``nv_humanoid.xml`` and Steel applies it as passive springs; the
    scene tests run the humanoid without them. The fixtures that load ``nv_humanoid.xml`` apply it to their models.
    """
    if model.joint_stiffness is not None:
        model.joint_stiffness.zero_()
    return model


def _humanoids(device, worlds=1, per_world=1, *, props=False, spacing=2.0, lying=True):
    """``worlds`` copies of ``per_world`` ``nv_humanoid.xml`` humanoids (27 DOFs each) on the ground.

    The humanoids of a world sit ``spacing`` [m] apart on x from x = 0; with ``props`` one free
    0.3 m box per world sits beside them, half a spacing before the first. ``lying`` turns each
    humanoid's root 90 degrees about x at z = 0.375 m, onto its right side: after
    :func:`newton.eval_fk` its lowest collision shape, the right lower-arm capsule, is 4.0 mm
    above the ground, inside the 5 mm ``rigid_gap``. Otherwise it stands with its root at
    z = 1.3 m (:func:`_humanoid`). The builders register Steel's custom attributes first. The
    contact capacity is 96 contacts per humanoid, twice that when humanoids share a world, plus
    32 per box. The MJCF's passive joint stiffness is zeroed (:func:`_without_passive_springs`).
    """
    humanoid = _humanoid(lying=lying)
    template = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverSteel.register_custom_attributes(template)
    template.rigid_gap = 0.005
    template.replicate(
        humanoid,
        per_world,
        separate_worlds=False,
        xforms=[wp.transform(wp.vec3(i * spacing, 0.0, 0.0), wp.quat_identity()) for i in range(per_world)],
    )
    if props:
        box = template.add_body(xform=wp.transform(wp.vec3(-0.5 * spacing, 0.0, 0.15), wp.quat_identity()))
        template.add_shape_box(box, hx=0.15, hy=0.15, hz=0.15)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverSteel.register_custom_attributes(builder)
    builder.rigid_gap = 0.005
    builder.replicate(template, worlds)
    builder.add_ground_plane()
    model = _without_passive_springs(builder.finalize(device=device))
    humanoids = worlds * per_world
    boxes = worlds if props else 0
    model.rigid_contact_max = 96 * humanoids * (2 if per_world > 1 else 1) + 32 * boxes
    return model


def _make_chain_world(n_links: int, root_height: float = 1.0, free_box: bool = False) -> newton.ModelBuilder:
    """A fixed-base serial chain of ``n_links`` revolute links (n_links DOFs), optionally with a free box."""
    builder = newton.ModelBuilder()
    joints = []
    prev = -1
    for _ in range(n_links):
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.15, hy=0.03, hz=0.03)
        if prev == -1:
            parent_xform = wp.transform(p=wp.vec3(0.0, 0.0, root_height), q=wp.quat_identity())
        else:
            parent_xform = wp.transform(p=wp.vec3(0.15, 0.0, 0.0), q=wp.quat_identity())
        joints.append(
            builder.add_joint_revolute(
                parent=prev,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=parent_xform,
                child_xform=wp.transform(p=wp.vec3(-0.15, 0.0, 0.0), q=wp.quat_identity()),
            )
        )
        prev = link
    builder.add_articulation(joints)
    if free_box:
        body = builder.add_body(xform=wp.transform(wp.vec3(-1.0, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    return builder


def _chain_worlds(world_link_counts: list[int], device, root_height: float = 1.0, free_box=None) -> newton.Model:
    """One chain world per entry in ``world_link_counts``, plus a ground plane."""
    scene = newton.ModelBuilder()
    for world, n_links in enumerate(world_link_counts):
        scene.add_world(_make_chain_world(n_links, root_height, free_box is not None and free_box[world]))
    scene.add_ground_plane()
    return scene.finalize(device=device)


_CHAIN_KIND = re.compile(r"chain:(?P<links>[1-9][0-9]*)(?:@(?P<height>[0-9.]+))?(?P<box>\+box)?")


def _siblings() -> newton.ModelBuilder:
    """A base fixed 1 m above the ground with three revolute children hanging from it (3 DOFs)."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    base = builder.add_link()
    builder.add_shape_box(base, hx=0.1, hy=0.1, hz=0.05)
    joints = [builder.add_joint_fixed(-1, base, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))]
    for x, axis in ((-0.15, newton.Axis.Y), (0.0, newton.Axis.X), (0.15, newton.Axis.Y)):
        child = builder.add_link()
        builder.add_shape_box(child, hx=0.03, hy=0.03, hz=0.15)
        joints.append(
            builder.add_joint_revolute(
                base,
                child,
                axis=axis,
                parent_xform=wp.transform(wp.vec3(x, 0.0, -0.05), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.15), wp.quat_identity()),
            )
        )
    builder.add_articulation(joints)
    return builder


def _scene_world(kind: str) -> newton.ModelBuilder:
    """The builder of one world of :func:`_scene_worlds` (see there for the kinds)."""
    if kind == "humanoid":
        return _humanoid()
    if kind == "pendulum":
        return _make_chain_world(1)
    if kind == "siblings":
        return _siblings()
    if kind == "box":
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        box = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.15), wp.quat_identity()))
        builder.add_shape_box(box, hx=0.15, hy=0.15, hz=0.15)
        return builder
    chain = _CHAIN_KIND.fullmatch(kind)
    if chain is None:
        raise ValueError(f"unknown scene world kind {kind!r}")
    height = float(chain["height"]) if chain["height"] else 1.0
    return _make_chain_world(int(chain["links"]), root_height=height, free_box=bool(chain["box"]))


def _scene_builder(kinds: Sequence[str], xforms: Sequence | None = None) -> newton.ModelBuilder:
    """The builder of :func:`_scene_worlds`, world ``i`` placed by ``xforms[i]``; open for global (world -1) bodies."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverSteel.register_custom_attributes(builder)
    builder.rigid_gap = 0.005
    for index, kind in enumerate(kinds):
        builder.add_world(_scene_world(kind), xform=None if xforms is None else xforms[index])
    builder.add_ground_plane()
    return builder


def _scene_worlds(device, kinds: Sequence[str]) -> newton.Model:
    """One world per entry of ``kinds`` on a ground plane, with Steel's custom attributes registered.

    The kinds:

    - ``"humanoid"``: the lying ``nv_humanoid.xml`` humanoid of :func:`_humanoids` (27 DOFs);
    - ``"chain:N"``: :func:`_make_chain_world` with N links, a fixed-base serial chain of N
      revolute links of 0.3 m along x, its root 1 m above the ground (N DOFs); ``"chain:N@H"``
      puts the root at height H [m], so ``@0.03`` lays the chain on the ground; a ``"+box"``
      suffix adds the world's free 0.2 m box (``free_box=True``);
    - ``"pendulum"``: ``"chain:1"``;
    - ``"siblings"``: a fixed base with three revolute children (3 DOFs);
    - ``"box"``: a free 0.3 m box on the ground.

    The ground plane and the humanoids have a 5 mm contact gap, the chains and their boxes
    ``ModelBuilder``'s default 0.1 m. The humanoids' passive joint stiffness is zeroed
    (:func:`_without_passive_springs`).
    """
    return _without_passive_springs(_scene_builder(kinds).finalize(device=device))


def _world_parts(model, world: int) -> tuple[np.ndarray, np.ndarray]:
    """``world``'s joint coordinates (indices into ``joint_q``) and its free rigid bodies (indices into ``body_q``)."""
    joints = np.flatnonzero(model.joint_world.numpy() == world)
    q_start = model.joint_q_start.numpy()
    coordinates = np.concatenate([np.arange(q_start[joint], q_start[joint + 1]) for joint in joints])
    art_start = model.articulation_start.numpy()
    single_joint = np.diff(art_start) == 1
    free_roots = model.joint_type.numpy()[art_start[:-1]] == int(newton.JointType.FREE)
    arts = np.flatnonzero(single_joint & free_roots & (model.articulation_world.numpy() == world))
    return coordinates, model.joint_child.numpy()[art_start[arts]]


def _leaning_humanoid(device, *, distance=0.75, tilt_deg=15.0):
    """A free drum on the ground and one ``nv_humanoid.xml`` humanoid that falls onto it and stays there.

    The drum is an upright cylinder 0.8 m across and 0.8 m tall, 51.2 kg,
    friction 0.8, standing at the origin. The humanoid stands as in :func:`_humanoids` (root at
    z = 1.3 m) ``distance`` [m] from the drum's axis on +x, facing the drum, tilted ``tilt_deg``
    toward it about its feet. Its passive springs are zeroed (:func:`_without_passive_springs`), so with zero
    control the humanoid is limp: it falls onto the drum within the first 10 control steps (of 5 steps of
    1/200 s) and lies across the drum's top with its feet on the ground to the end, so the drum
    carries part of its weight and a sideways push. Returns the model and the drum's body index.
    """
    humanoid = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverSteel.register_custom_attributes(humanoid)
    humanoid.rigid_gap = 0.005
    humanoid.add_mjcf(
        newton.examples.get_asset("nv_humanoid.xml"),
        xform=wp.transform(wp.vec3(0.0, 0.0, 1.3), wp.quat_identity()),
        ignore_names=["floor", "ground"],
        up_axis="Z",
    )
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverSteel.register_custom_attributes(builder)
    builder.rigid_gap = 0.005
    # The humanoid faces +x: turn it to face the drum, then tilt it forward about its feet.
    facing = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi)
    tilt = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(np.radians(tilt_deg)))
    builder.add_builder(humanoid, xform=wp.transform(wp.vec3(distance, 0.0, 0.0), facing * tilt))
    radius, height, mass = 0.4, 0.8, 51.2
    drum = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 * height), wp.quat_identity()))
    builder.add_shape_cylinder(
        drum,
        radius=radius,
        half_height=0.5 * height,
        cfg=newton.ModelBuilder.ShapeConfig(density=mass / (np.pi * radius * radius * height), mu=0.8),
    )
    builder.add_ground_plane()
    model = _without_passive_springs(builder.finalize(device=device))
    # As _humanoids: 96 contacts for the humanoid, twice that as it shares its world, 32 for the drum.
    model.rigid_contact_max = 2 * 96 + 32
    return model, drum


def row_family_arrays(solver) -> list[tuple[wp.array, wp.array, wp.array]]:
    """Each row family's ``(count, row_type, impulses)`` arrays, indexed by world.

    Steel's dense rows come first; they hold every row of the scene solve. Steel's free-body rows follow, then its
    propagation rows when the propagation responses are active; the scene solve leaves the free-body rows empty and
    builds no propagation rows.
    """
    families = [
        (solver._constraint_count, solver._row_type, solver._impulses),
        (solver._mf_constraint_count, solver._mf_row_type, solver._mf_impulses),
    ]
    if solver._propagation_active:
        families.append(
            (solver._propagation_constraint_count, solver._propagation_row_type, solver._propagation_impulses)
        )
    return families


def row_families(solver, world: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """``world``'s ``(row_type, impulses)`` of each row family (:func:`row_family_arrays`), cut at its row count."""
    families = []
    for count, row_type, impulses in row_family_arrays(solver):
        rows = int(count.numpy()[world])
        families.append((row_type.numpy()[world, :rows], impulses.numpy()[world, :rows]))
    return families


SCENE = {
    "solve_mode": "scene",
    "iterations": 4,
    "enable_joint_limits": True,
    "joint_limit_activation_gap": 0.1,
    "friction_anchor_beta": 0.0,
}
"""The scene tests' options: the scene solve with four iterations, joint limits and point friction."""


def _steel_on(model, options: dict) -> dict:
    """``options`` of a solver on ``model``, returned as they are.

    A Steel reference (``solve_mode="serial"``) runs its serial solve on either device, so no device needs a
    substitute solve. The scene tests pass their options through here, the one place a reference's options would
    be adapted to ``model``'s device.
    """
    return options


class _Run:
    """A scene solver stepping its own states with its own deterministic collision pipeline.

    ``options`` replace the :data:`SCENE` options (``solve_mode="serial"`` makes it Steel's serial solver,
    :func:`_steel_on`). ``collide=False`` steps without contacts.
    """

    def __init__(self, model, rows, overrides, *, collide=True, **options):
        with kernel_overrides(overrides):
            self.solver = SolverSteel(model, articulation_constraint_max=rows, **_steel_on(model, {**SCENE, **options}))
        self.pipeline = newton.CollisionPipeline(model, deterministic=True) if collide else None
        self.contacts = self.pipeline.contacts() if collide else None
        self.state_0 = model.state()
        self.state_1 = model.state()
        self.control = model.control()
        newton.eval_fk(model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        if self.solver.solve_mode == "scene":
            # A row's free response is written only over its free prefix; the same fill makes the rest comparable.
            self.solver._sparse_row_free_response.fill_(-7.0)

    def step(self):
        if self.pipeline is not None:
            self.pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, DT)
        self.state_0, self.state_1 = self.state_1, self.state_0


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    """Root mean square of ``a - b`` in float64."""
    return float(np.sqrt(np.mean((np.asarray(a, dtype=np.float64) - b) ** 2)))


def _global_box_model(device):
    """One box on the ground replicated into two worlds at the same pose, and a global box resting on both."""
    scene = newton.ModelBuilder()
    body = scene.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()), mass=1.0)
    scene.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder = newton.ModelBuilder()
    builder.replicate(scene, 2)
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.3), wp.quat_identity()), mass=1.0)
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    model.rigid_contact_max = CONTACT_MAX
    return model


REBOUND_DROP = 0.05
"""Height [m] of the sphere of the rebound tests above the plane: it strikes at about 1 m/s."""

REBOUND_GRAVITY = (0.0, 0.0, -9.81)

REBOUND_STEPS = 60
"""Steps of the rebound tests at :data:`DT`: the fall and two rebounds with their flights."""

REBOUND_ROWS = 32
"""``articulation_constraint_max`` of the rebound tests (one contact, three rows)."""

REBOUND_RTOL = 1.0e-4
"""A scene velocity agrees with Steel's within this fraction of Steel's magnitude."""

REBOUND_APEX_ATOL = 1.0e-6
"""Absolute floor [m/s] of the rebound tests' velocity comparison, applied at every step.

:func:`assert_rebound_tracks_steel` adds it to the relative tolerance at each step, which matters
where the velocity crosses zero (an apex) and the relative tolerance vanishes. A rebound target is Steel's
bit for bit, ``-target - rebound`` with the rebound velocity ``-e * u_start`` of the same step-start closing
speed (``kernels.contact_rebound_velocity``, which the scene's and Steel's row builders share); what the
floor covers is the response the target is solved through. The articulated sphere's rows take their
response from the sparse factor, Steel's from its dense rows, and the impulse that meets the target moves
the sphere by a velocity rounded apart (the retired incident-speed law measured 2 ulp, 6e-8 m/s, at an apex
where Steel's velocity is 7e-8 m/s; to be re-measured once the scene mode steps). With velocity iterations
(``test_velocity_iterations_rebound_tracks_steel``) the free sphere takes this floor too: its velocity pass
sums the row's ``J v`` from the incident velocity and ``delta``, where Steel's free-body rows dot their
Jacobian with the velocity, 1 ulp apart (6e-8 m/s at an apex where Steel's velocity is 4e-8 m/s).
"""


def _drop_sphere(device, *, restitution, scene="free", notify_after=None):
    """Drop the sphere of ``_build_plane_model`` onto the plane with a scene solver and Steel's matrix-free solver.

    Returns the two runs (scene first) and the sphere's vertical velocity after each step, one row per
    run. With ``notify_after`` the model is built without restitution, and the coefficients become
    ``restitution`` through a ``SHAPE_PROPERTIES`` notify before that step.
    """
    authored = restitution if notify_after is None else 0.0
    model, body = _build_plane_model(
        device, separation=REBOUND_DROP, restitution=authored, gravity=REBOUND_GRAVITY, scene=scene
    )
    runs = (_Run(model, REBOUND_ROWS, {}), _Run(model, REBOUND_ROWS, {}, solve_mode="serial"))
    velocities = np.zeros((len(runs), REBOUND_STEPS), dtype=np.float32)
    for step in range(REBOUND_STEPS):
        if step == notify_after:
            model.shape_material_restitution.fill_(restitution)
            for run in runs:
                run.solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        for k, run in enumerate(runs):
            run.step()
            velocities[k, step] = run.state_0.body_qd.numpy()[body, 2]
    return runs, velocities


def assert_rebound_tracks_steel(test, runs, velocities, label="", atol=0.0):
    """Check that the sphere rebounds after an impact faster than the threshold, as Steel's does, at every step.

    The scene's vertical velocity agrees with Steel's within :data:`REBOUND_RTOL` of Steel's magnitude,
    plus ``atol``, at every step.
    """
    scene_run, steel_run = runs
    test.assertEqual(scene_run.solver.solve_mode, "scene", label)
    test.assertIsNone(steel_run.solver._scene, label)
    scene, steel = velocities
    rebound = int(np.argmax(steel > 0.0))
    test.assertGreater(rebound, 0, f"{label}: Steel's sphere never rebounds")
    test.assertLess(steel[rebound - 1], -steel_run.solver.restitution_velocity_threshold, label)
    test.assertGreater(scene[rebound], 0.0, f"{label}: the scene's sphere does not rebound")
    np.testing.assert_allclose(scene, steel, rtol=REBOUND_RTOL, atol=atol, err_msg=label)


# scene_pgs: the default scene kernels' hashes

DEFAULT_SNIPPET_SHA256: dict[str, str] = {
    "21/False/response": "ec4be98c37f0472d3732b02bc4e3c9c542166e637882bc8f65ed4fd79708a3b8",
    "21/False/sides": "a7044ccdd33238488b6fd7d3d80ed2b5c9dbd5bd99ddfb11c35c2b6377515717",
    "21/False/serial": "f6f99bb7535380233ef006f8fd8b4f41d6cbac50ec5af4113fa960ff7b3ed6c7",
    "21/False/order_serial": "ee903605980b90591544724b91a7dfe4f7f876c136a0641907c5b0a78cfff345",
    "21/False/color": "8b53b2a0f00b890131bbf9419492dcb3a882dfa63e1d3005a02484284c932b01",
    "21/False/block": "df41b9e6aeff7ea08b62d41f1d17325bf092f6eb88326db31f6982004d38d7e1",
    "30/False/response": "ec4be98c37f0472d3732b02bc4e3c9c542166e637882bc8f65ed4fd79708a3b8",
    "30/False/sides": "68ec644078a5661feea0878b20e8394d07fd0fd9fea63d17f87243a686dc0ad1",
    "30/False/serial": "3dd994d6e4e1015d74eaa70ac25973debb01e8a3b3f885f6c49400a77097fa26",
    "30/False/order_serial": "1890a16108be081847e6d4125c2c029c70a72e4ec8887be48b6119d52a843a48",
    "30/False/color": "2c87549e875242daa61b7178ecb650ba40661c53be2bc22d730702ee567edd5c",
    "30/False/block": "a40816654e1e7e7700256d3cf62fcf7b15562093f2da8c5a51dce5061896899e",
    "12/True/response": "b581394683041fab33354fd795767bbba058196b8a3991b99c5d2fc114c16c0f",
    "12/True/sides": "4e128591b2fcac2895457c6fe1b5c787db92957c63af195cd66c5542a0684cef",
    "12/True/serial": "240f01468759b6ed31e151ef928291477b7dc5026083dd48ea10b9e3b8c9625d",
    "12/True/order_serial": "2aad0922add4631bb18a30cbce109cf3993be43a74b1b2bddfb962cc306b79da",
    "12/True/color": "d4437d281a3de2a91e195d9f4beff49bd75f5503142ea9bcdc913ae5271f0b34",
    "12/True/block": "fe909a2cdbdbaa909d216842cb6b823729f530ea45a9fea017e709ac1d8781f6",
    "sparse_contact/27": "2d4ce7c5be47447f2463a1a302e4e1e52efb65f0f215f00d60cc234d67179bff",
}
"""SHA-256 of the native snippets of the default scene kernels and of Steel's sparse contact response snippet.

A ``"<support>/<free_response>/<kernel>"`` key hashes one snippet of ``scene_pgs.scene_kernel_snippets(support,
free_response=free_response, dof_max=dof_max)`` for the configurations (21, False, 27), (30, False, 27) and
(12, True, 6): the group-table contact response (``response``) and the sweep snippets (``sides``, ``serial``,
``order_serial``, ``color``, ``block``) with ``"lanes8"`` sides and units, Steel-order sums and the ``"BA"`` order
oracle. ``sparse_contact/27`` hashes ``scene_rows._response_snippet(27, specialized=True)``, which is the snippet of
``_get_sparse_contact_response_kernel(27)``, Steel's own text, and equals that entry of :data:`STEEL_TEXT_SHA256`.

Recorded from ``scene_pgs`` as it re-derived the sweep from Steel's current text. Against the text before, the
fifteen sweep entries moved and nothing else: the line-number citations in the snippets' comments became names, and
the thread loop's tree macro was renamed; the code of every default snippet is unchanged. The ``response`` entries and
``sparse_contact/27`` did not move. The tree macro is now ``SCENE_TREE_LEVEL``, so that no name of the solver package
holds ``STEEL_`` (``test_options.test_solver_reads_no_environment_variables``): the snippets with the one-thread loop
(``serial``, ``order_serial`` and ``block`` of each configuration) were re-recorded for that rename alone.
``test_scene_tripwire.test_scene_kernels_are_pinned`` checks the table: a change to a default scene kernel's source
must change it on purpose.
"""


STEEL_TEXT_SHA256: dict[str, str] = {
    "mf_gs/128/64/27": "6e452fdcc67b374aec9ecadc7a0bcadc0d627c0f9de155c9a26aa005ea4fbe9a",
    "mf_gs/128/64/27/nesterov": "47914e38f6e90adf9a43a88c82571f0c30fef912024e6e482a70d9002393bd85",
    "sparse_contact/27": "2d4ce7c5be47447f2463a1a302e4e1e52efb65f0f215f00d60cc234d67179bff",
    "joint_limit_rows/1": "1be38e441e3d224d07a1245b2f0e6976e57ee2a96da629a10911a38bb6f185ed",
    "torsion_sweep/12": "9a85d072305818a1306d3088f8f87d4b7f8d2f27b72413085662a6de59c53f23",
    "torsion_sweep/12/linked": "08d9a84e12af3e5aeae1622404604dc119a6022b605e20f5caeb600254990d71",
    "friction_pair": "9f644efe8ccaa94e76b53bd91922289805c95cbb5da2b13f6dd70d5e1d85b0f8",
    "dense_bisection_friction": "3b6194aaf540de2f9270a55e3000ddf135b9fa79d3ea5526d68c6e70fdd5230d",
    "mf_bisection_friction": "664695a204de43ff00d2f47c5fcc7153282673a88d1303288dbb76e7c34f8827",
    "mf_desaxce_bias": "2a0004080d1df6c9e9c1a980ef0916cb82414509d2ac070e90a8c0b18372e675",
    "mf_coulomb_newton_friction": "38ee856a3db591aed7644f5c926476a20cc009d45168b34d0ecaaec70c32cb7b",
    "compliance_prepare_contacts": "a0efc8a979580b343faed818656b42727106a57f2142229a8f7bdba18e0e8c5d",
}
"""SHA-256 of the default path's kernel text, keyed as ``test_scene_tripwire.steel_kernel_texts`` names it.

The scene mode reuses Steel's laws by import or keeps copies that guard tests compare with Steel's source, so a
change to one of these texts is a change the scene mode has to follow. A change to a text must change this table on
purpose: re-record it (``python -m newton.tests.steel.test_scene_tripwire --record``) with the reason in the same
commit.
"""


# scene_units: the scene's units built on a solver without the scene mode


class _WithoutSceneMode:
    """``solver`` as the scene's planning functions read a solver without the scene mode (``_scene_mode`` false)."""

    _scene_mode = False

    def __init__(self, solver):
        self._solver = solver

    def __getattr__(self, name):
        return getattr(self._solver, name)


def _family_bearing(solver) -> np.ndarray:
    """The row families each articulation of ``solver`` can bear: bit ``1 << family``, ``int32``, one per articulation.

    The scene mode's per-articulation flags (``_scene_family_bearing``, :func:`scene_internal_bearing`) read on a
    solver without it: drives with ``drive_mode="rows"`` on a PRISMATIC, REVOLUTE or D6 DOF, joint limits with
    ``enable_joint_limits`` on such a DOF with a finite bound, joint friction with ``enable_joint_friction`` on a DOF
    whose joint supports friction. Mimic rows, loop closures and velocity limits are the scene solve's rows, so no
    articulation bears them here: the velocity-limit bit is cleared.
    """
    bearing = scene_internal_bearing(_WithoutSceneMode(solver), solver.model)
    return bearing & ((1 << FAMILY_VELOCITY_LIMIT) - 1)


class _UnitsView:
    """A solver without the scene mode as :class:`SceneUnits` reads it: its attributes and the scene mode's two flags.

    ``_scene_family_bearing`` is :func:`_family_bearing` and ``_scene_late_rows_enabled`` is ``False``: late rows are
    the scene solve's, so units built without it have no late unit.
    """

    def __init__(self, solver):
        self._solver = solver
        self._scene_family_bearing = _family_bearing(solver)
        self._scene_late_rows_enabled = False

    def __getattr__(self, name):
        return getattr(self._solver, name)


def _units_view(solver) -> _UnitsView:
    """``solver``, a solver without the scene mode, as :class:`SceneUnits` reads it at construction."""
    return _UnitsView(solver)


def _units_for(solver, contacts, unit_max: int | None = None) -> SceneUnits:
    """The scene units of ``solver``'s last step: built from its contact routing on ``contacts``, then colored.

    ``solver`` is a solver without the scene mode that has taken one step on ``contacts``, so its contacts are routed
    (``_contact_path``, ``_contact_art_a/b``, ``_contact_slots_needed``). The units take their color cap and pair rule
    from the solver's kernel overrides (``scene_color_cap``, ``scene_pair_rule``) and their capacity from
    ``unit_max``, else from the build. Without row counts, an articulation's internal unit exists when it bears joint
    limits and holds no rows.
    """
    units = SceneUnits(_units_view(solver), unit_max)
    units.build(solver, contacts)
    units.color()
    return units


# scene_rows: the internal-unit family hashes

LIMIT_ROWS_SNIPPET_SHA256 = {
    1: "1be38e441e3d224d07a1245b2f0e6976e57ee2a96da629a10911a38bb6f185ed",
    2: "8d8ab3bc6edfcbcbe8eb9565f56fd7a0e0cad32dc62a235ef20c1e5f325187c9",
}
"""SHA-256 of ``scene_rows.sparse_joint_limit_rows_snippet(mask_words)``, the joint-limit family's expansion of the
internal-row template. The one-word hash is Steel's own builder's text, ``STEEL_TEXT_SHA256["joint_limit_rows/1"]``."""

FRICTION_ROWS_SNIPPET_SHA256 = {
    1: "622fc7e13c428c745fac15579c9c80eeb5893f2fd804c2b7769247c06d0676a5",
    2: "3bcdad2341a30a3ec1c1da6bf06f31ae6245161edef489fbc286c7992970fa03",
}
"""SHA-256 of ``scene_rows.sparse_joint_friction_rows_snippet(mask_words)``, the joint-friction family's expansion."""

DRIVE_ROWS_SNIPPET_SHA256 = {
    1: "6949cfb63e42a400b8b4540e3ac89e26049a00b6cf9d2b72ae1fc11ceee9fb8f",
    2: "0f579a2623470681cd25bbbfcd7055a77f7315355f4f09991ace11e6ec5344b5",
}
"""SHA-256 of ``scene_rows.sparse_drive_rows_snippet(mask_words)``, the drive family's expansion."""

FUSED_DRIVE_ROWS_SNIPPET_SHA256 = {
    1: "785aaa59cd24c15ff8de1154ca2beea10a45566291e4ac344a82959cebcdcf63",
    2: "e5139dc4b643912976486aae6e1eca4d68bf31a96ad7091598392db07f2b7fbd",
}
"""SHA-256 of ``scene_rows.sparse_drive_rows_snippet(mask_words, fused=True)``, the drive family's expansion with the
fused velocity clamp's row tail."""

VELOCITY_LIMIT_ROWS_SNIPPET_SHA256 = {
    1: "ee33027949e01772d12908033acc719d42d8e3748a29cc1a1da6e40bcbde023c",
    2: "749782aa3890db926c1bd6acd1962c2455e944d7520bf88043e6fc96797b1238",
}
"""SHA-256 of ``scene_rows.sparse_joint_velocity_limit_rows_snippet(mask_words)``, the velocity-limit family's
expansion."""

MIMIC_ROWS_SNIPPET_SHA256 = {
    1: "6fc618e56aa17820d621bf9b152f19aba4cd8832ff03f270da7ec480ead8d51a",
    2: "90530fb5b90c6318a510756d896910f7870a6447fbda9be6dfaa9e3ca66459ab",
}
"""SHA-256 of ``scene_rows.sparse_mimic_rows_snippet(mask_words)``, the mimic family's expansion with its support-mask
load.

The six tables pin the row families' builder text. All six were re-recorded when the template was re-cut from Steel's
current joint-limit builder, whose reservation comment names ``reserve_world_row_slots``; that comment is the only
change to every family's text. A change to a family's builder must change its table on purpose."""

# sparse_mass_matrix: the sparse mass-factor kernel at one nv_humanoid.xml humanoid's factor plan
STEEL_TEXT_SHA256["factor_kernel"] = "6378a91d66938b3fb79213e8ac5f1b09fbbef2aa9dfb502863b2e2e41e8f93c7"


# scene_warmstart and scene_solve: the heap of the scene tests


def _scene_heap(
    device,
    nx=6,
    ny=6,
    nz=3,
    pitch=0.042,
    half=0.02,
    *,
    worlds=1,
    register_attributes=False,
    spare=False,
) -> newton.Model:
    """``_fixtures._heap`` (a dense heap of touching boxes on the ground) for the scene tests, in ``worlds`` worlds.

    With ``register_attributes`` the builder registers Steel's custom attributes
    (:meth:`~newton.solvers.SolverSteel.register_custom_attributes`) before any body is added. With ``spare`` one more
    box, the last body, rests on the ground 0.5 m beside the heap, clear of it. With ``worlds > 1`` the heap with its
    ground is replicated into that many worlds.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    if register_attributes:
        SolverSteel.register_custom_attributes(builder)
    builder.rigid_gap = 0.002
    builder.add_ground_plane()
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                body = builder.add_body(
                    xform=wp.transform(
                        wp.vec3((ix - nx / 2) * pitch, (iy - ny / 2) * pitch, 0.021 + iz * (2 * half + 0.001)),
                        wp.quat_identity(),
                    )
                )
                builder.add_shape_box(body, hx=half, hy=half, hz=half)
    if spare:
        body = builder.add_body(xform=wp.transform(wp.vec3(0.5, 0.0, half + 0.001), wp.quat_identity()))
        builder.add_shape_box(body, hx=half, hy=half, hz=half)
    if worlds > 1:
        heap = builder
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.replicate(heap, copy_count=worlds)
    model = builder.finalize(device=device)
    # Sizes the solver's contact scratch, so it is set before the solver is built.
    model.rigid_contact_max = CONTACT_MAX
    return model
