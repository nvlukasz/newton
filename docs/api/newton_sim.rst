newton.sim
==========

.. currentmodule:: newton.sim

.. rubric:: Submodules

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ik

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   CollisionPipeline
   Contacts
   Control
   Model
   ModelBuilder
   State
   Style3DModel
   Style3DModelBuilder

.. rubric:: Functions

.. autosummary::
   :toctree: _generated
   :signatures: long

   color_graph
   count_rigid_contact_points
   eval_fk
   eval_ik
   get_joint_dof_count
   plot_graph

.. rubric:: Constants

.. list-table::
   :header-rows: 1

   * - Name
     - Value
   * - JointType.BALL
     - 2
   * - JointType.D6
     - 6
   * - JointType.DISTANCE
     - 5
   * - JointType.FIXED
     - 3
   * - JointType.FREE
     - 4
   * - JointMode.NONE
     - 0
   * - JointMode.TARGET_POSITION
     - 1
   * - JointMode.TARGET_VELOCITY
     - 2
   * - JointType.PRISMATIC
     - 0
   * - JointType.REVOLUTE
     - 1
   * - SolverNotifyFlags.BODY_INERTIAL_PROPERTIES
     - 8
   * - SolverNotifyFlags.BODY_PROPERTIES
     - 4
   * - SolverNotifyFlags.JOINT_DOF_PROPERTIES
     - 2
   * - SolverNotifyFlags.JOINT_PROPERTIES
     - 1
   * - SolverNotifyFlags.SHAPE_PROPERTIES
     - 16
