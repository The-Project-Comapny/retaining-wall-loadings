"""
Author: Mishal Mohanlal
Date: 31/08/2026 (first working model)

"""


import ast
import heapq
import math
from collections import Counter
from pathlib import Path

import gmsh
import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops
import pandas as pd
from matplotlib.patches import FancyArrowPatch


# ============================================================
# User inputs
# ============================================================

STP_LOCATION = (
    r"C:\Users\MishalMohanlal\OneDrive - The Project Company\Documents"
    r"\GitHub\retaining wall loadings\TIB BIN RAMP FEA 1.stp"
)

MSH_LOCATION = (
    r"C:\Users\MishalMohanlal\OneDrive - The Project Company\Documents"
    r"\GitHub\retaining wall loadings\TIB BIN RAMP FEA 1.msh"
)

# The complete model uses N, mm and seconds.
SOIL_E_KPA = 30_000.0
SOIL_POISSONS_RATIO = 0.30
SOIL_DENSITY_MG_M3 = 1.8
SOIL_FRICTION_ANGLE_DEG = 32.0
SOIL_COHESION_KPA = 1.0
SOIL_DILATION_ANGLE_DEG = 0.0
ATMOSPHERIC_PRESSURE_KPA = 101.0
SOIL_HARDENING_RATIO = 1.0

MODEL_WIDTH_MM = 1000.0
BEAM_E_N_MM2 = 30_000.0
BEAM_THICKNESS_MM = 300.0
BEAM_WIDTH_MM = MODEL_WIDTH_MM

GRAVITY_M_S2 = 9.81
NUMBER_OF_GRAVITY_STEPS = 100

# Point loads are applied after gravity has converged and been held
# constant. Each entry is:
#
#     (node_tag, signed_magnitude, direction)
#
# Directions: "x", "z", or "moment". Forces use N; moments use N.mm.
# Leave this list empty to run gravity only. Examples:
#
# POINT_LOADS = [
#     (7, -10_000.0, "z"),
#     (52, 5_000.0, "x"),
#     (52, 2.0e6, "moment"),
# ]
POINT_LOADS = [
    (7, -100_000.0, "z"),       # 100 kN downward
    # (52, 5_000.0, "x"),        # 5 kN right     # 2 kN.m counter-clockwise
]
NUMBER_OF_POINT_LOAD_STEPS = 100
POINT_LOAD_TIME_SERIES_TAG = 2
POINT_LOAD_PATTERN_TAG = 2

SHOW_RESULTS_PLOT = True
SAVE_RESULTS_PLOT = True
RESULTS_PLOT_PATH = Path(__file__).resolve().with_name(
    "retaining_wall_force_diagrams.png"
)
SHOW_NODE_TAGS_ON_PLOT = True

# Set this to False to reuse an existing .msh file.
RECREATE_MESH = True

# Set this to True if the Gmsh GUI should open after meshing.
SHOW_GMSH_GUI = True


# ============================================================
# Unit conversions: N-mm-s
# ============================================================

KPA_TO_N_MM2 = 1.0e-3
MG_M3_TO_N_S2_MM4 = 1.0e-9
M_TO_MM = 1000.0


def create_mesh(
    stp_path,
    msh_path,
    mesh_size_min=100.0,
    mesh_size_max=1000.0,
    show_gui=True,
):
    """Create a quadrilateral surface mesh and write a Gmsh 2.2 file."""
    # GUI analyses run this function from a QThread. Gmsh's default Python
    # initializer installs a SIGINT handler, which Python only permits on the
    # main thread. Disabling that optional handler keeps both CLI and threaded
    # execution valid.
    gmsh.initialize(interruptible=False)

    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("retaining_wall_soil")
        gmsh.model.occ.importShapes(stp_path)
        gmsh.model.occ.synchronize()

        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_min)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_max)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.recombine()
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        gmsh.model.mesh.refine()

        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(msh_path)

        if show_gui:
            gmsh.fltk.run()
    finally:
        gmsh.finalize()


def read_msh_to_dataframe(msh_path):
    """Read Gmsh nodes and elements into one normalized DataFrame."""
    # See create_mesh(): do not register Python signal handlers from a GUI
    # worker thread.
    gmsh.initialize(interruptible=False)

    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.open(msh_path)
        records = []

        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()

        for index, node_tag in enumerate(node_tags):
            records.append(
                {
                    "record_type": "node",
                    "tag": int(node_tag),
                    "x": float(coordinates[3 * index]),
                    "y": float(coordinates[3 * index + 1]),
                    "z": float(coordinates[3 * index + 2]),
                    "entity_dimension": None,
                    "entity_tag": None,
                    "element_type": None,
                    "element_name": None,
                    "element_order": None,
                    "connectivity": None,
                }
            )

        for dimension, entity_tag in gmsh.model.getEntities():
            element_types, element_tags, node_tags_by_type = (
                gmsh.model.mesh.getElements(dimension, entity_tag)
            )

            for element_type, tags, flat_connectivity in zip(
                element_types,
                element_tags,
                node_tags_by_type,
            ):
                (
                    element_name,
                    _,
                    element_order,
                    nodes_per_element,
                    _,
                    _,
                ) = gmsh.model.mesh.getElementProperties(element_type)

                for index, element_tag in enumerate(tags):
                    start = index * nodes_per_element
                    end = start + nodes_per_element

                    records.append(
                        {
                            "record_type": "element",
                            "tag": int(element_tag),
                            "x": None,
                            "y": None,
                            "z": None,
                            "entity_dimension": int(dimension),
                            "entity_tag": int(entity_tag),
                            "element_type": int(element_type),
                            "element_name": element_name,
                            "element_order": int(element_order),
                            "connectivity": [
                                int(tag)
                                for tag in flat_connectivity[start:end]
                            ],
                        }
                    )

        return pd.DataFrame(records)
    finally:
        gmsh.finalize()


def _parse_connectivity(value):
    """Return element connectivity as a list of integer node tags."""
    if isinstance(value, str):
        value = ast.literal_eval(value)
    elif isinstance(value, np.ndarray):
        value = value.tolist()

    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Invalid element connectivity: {value!r}")

    return [int(tag) for tag in value]


def _node_coordinate_map(mesh_df):
    node_df = mesh_df.loc[mesh_df["record_type"].eq("node")].copy()
    node_df["tag"] = node_df["tag"].astype(int)

    if node_df.empty:
        raise ValueError("No node records were found in mesh_df.")
    if node_df["tag"].duplicated().any():
        raise ValueError("Duplicate node tags were found in mesh_df.")

    return {
        int(row["tag"]): (
            float(row["x"]),
            float(row["y"]),
            float(row["z"]),
        )
        for _, row in node_df.iterrows()
    }


def _surface_quad_rows(mesh_df):
    """Return only first-order, four-node surface quadrilaterals."""
    element_df = mesh_df.loc[mesh_df["record_type"].eq("element")].copy()

    if element_df.empty:
        raise ValueError("No element records were found in mesh_df.")

    rows = []
    for _, row in element_df.iterrows():
        connectivity = _parse_connectivity(row["connectivity"])
        dimension = row.get("entity_dimension")
        element_type = row.get("element_type")

        is_surface = pd.isna(dimension) or int(dimension) == 2
        is_linear_quad = pd.isna(element_type) or int(element_type) == 3

        if is_surface and is_linear_quad and len(connectivity) == 4:
            rows.append((int(row["tag"]), connectivity))

    if not rows:
        raise ValueError("No first-order four-node surface quads were found.")

    return rows


def _signed_area(connectivity, planar_coordinates):
    points = [planar_coordinates[tag] for tag in connectivity]
    return 0.5 * sum(
        points[index][0] * points[(index + 1) % 4][1]
        - points[(index + 1) % 4][0] * points[index][1]
        for index in range(4)
    )


def create_opensees_sspquad_model(
    mesh_df,
    youngs_modulus,
    poissons_ratio,
    density,
    friction_angle,
    cohesion,
    dilation_angle=0.0,
    thickness=1000.0,
    material_tag=1,
    coordinate_columns=("x", "z"),
    body_force=(0.0, 0.0),
    atmospheric_pressure=0.101,
    hardening_ratio=0.0,
    material_model="drucker_prager",
):
    """Create a plane-strain SSPquad Drucker-Prager soil model."""
    material_key = str(material_model).strip().lower().replace("-", "_")
    material_aliases = {
        "drucker_prager": "drucker_prager",
        "druckerprager": "drucker_prager",
        "dp": "drucker_prager",
    }
    if material_key not in material_aliases:
        raise ValueError("material_model must be 'drucker_prager'.")
    material_key = material_aliases[material_key]

    if youngs_modulus <= 0.0:
        raise ValueError("youngs_modulus must be greater than zero.")
    if not (-1.0 < poissons_ratio < 0.5):
        raise ValueError("poissons_ratio must satisfy -1 < nu < 0.5.")
    if density < 0.0:
        raise ValueError("density cannot be negative.")
    if cohesion <= 0.0:
        raise ValueError(
            "DruckerPrager cohesion must be positive for this model. "
            "Use a small documented regularization value and perform "
            "a sensitivity study for nominally cohesionless soil."
        )
    if thickness <= 0.0:
        raise ValueError("thickness must be greater than zero.")
    if (
        not (0.0 <= dilation_angle <= friction_angle < 90.0)
    ):
        raise ValueError(
            "Require 0 <= dilation_angle <= friction_angle < 90 degrees."
        )
    if hardening_ratio < 0.0:
        raise ValueError("hardening_ratio cannot be negative.")
    if len(coordinate_columns) != 2:
        raise ValueError("coordinate_columns must contain two axes.")

    axis_1, axis_2 = coordinate_columns
    if axis_1 == axis_2 or {axis_1, axis_2} - {"x", "y", "z"}:
        raise ValueError("coordinate_columns must be two different x/y/z axes.")

    coordinates_3d = _node_coordinate_map(mesh_df)
    planar_coordinates = {
        tag: (
            coordinates[{"x": 0, "y": 1, "z": 2}[axis_1]],
            coordinates[{"x": 0, "y": 1, "z": 2}[axis_2]],
        )
        for tag, coordinates in coordinates_3d.items()
    }

    quads = []
    for element_tag, connectivity in _surface_quad_rows(mesh_df):
        missing = [tag for tag in connectivity if tag not in planar_coordinates]
        if missing:
            raise ValueError(
                f"Element {element_tag} references undefined nodes {missing}."
            )
        if len(set(connectivity)) != 4:
            raise ValueError(f"Element {element_tag} has repeated nodes.")

        area = _signed_area(connectivity, planar_coordinates)
        if np.isclose(area, 0.0):
            raise ValueError(
                f"Element {element_tag} has zero projected area in "
                f"the {coordinate_columns} plane."
            )
        if area < 0.0:
            connectivity = [
                connectivity[0],
                connectivity[3],
                connectivity[2],
                connectivity[1],
            ]
            area = -area

        quads.append((element_tag, connectivity, area))

    used_node_tags = sorted(
        {node_tag for _, connectivity, _ in quads for node_tag in connectivity}
    )

    bulk_modulus = youngs_modulus / (
        3.0 * (1.0 - 2.0 * poissons_ratio)
    )
    shear_modulus = youngs_modulus / (
        2.0 * (1.0 + poissons_ratio)
    )

    phi = math.radians(friction_angle)
    psi = math.radians(dilation_angle)

    friction_parameter = (
        2.0 * math.sqrt(2.0) * math.sin(phi)
        / (math.sqrt(3.0) * (3.0 - math.sin(phi)))
    )
    yield_stress = (
        6.0 * cohesion * math.cos(phi)
        / (math.sqrt(2.0) * (3.0 - math.sin(phi)))
    )
    dilation_parameter = (
        2.0 * math.sqrt(2.0) * math.sin(psi)
        / (math.sqrt(3.0) * (3.0 - math.sin(psi)))
    )

    # H = hardening_ratio * G. This changes the post-yield constitutive
    # response and must be selected explicitly, not treated as a solver
    # setting.
    hardening_modulus = hardening_ratio * shear_modulus

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)

    for node_tag in used_node_tags:
        x_model, y_model = planar_coordinates[node_tag]
        ops.node(node_tag, x_model, y_model)

    ops.nDMaterial(
        "DruckerPrager",
        int(material_tag),
        float(bulk_modulus),
        float(shear_modulus),
        float(yield_stress),
        float(friction_parameter),
        float(dilation_parameter),
        0.0,  # Kinf
        0.0,  # K0
        0.0,  # delta1
        0.0,  # delta2
        float(hardening_modulus),
        1.0,  # all optional hardening is isotropic
        float(density),
        float(atmospheric_pressure),
    )
    material_type = "DruckerPrager"

    b1, b2 = map(float, body_force)
    for element_tag, connectivity, _ in quads:
        ops.element(
            "SSPquad",
            element_tag,
            *connectivity,
            int(material_tag),
            "PlaneStrain",
            float(thickness),
            b1,
            b2,
        )

    return {
        "number_of_nodes": len(used_node_tags),
        "number_of_elements": len(quads),
        "node_tags": used_node_tags,
        "element_tags": [tag for tag, _, _ in quads],
        "element_connectivity": {
            tag: connectivity for tag, connectivity, _ in quads
        },
        "total_planar_area": sum(area for _, _, area in quads),
        "thickness": float(thickness),
        "material_tag": int(material_tag),
        "material_type": material_type,
        "material_model": material_key,
        "element_type": "SSPquad",
        "bulk_modulus": bulk_modulus,
        "shear_modulus": shear_modulus,
        "yield_stress": yield_stress,
        "friction_parameter": friction_parameter,
        "dilation_parameter": dilation_parameter,
        "hardening_modulus": hardening_modulus,
        "body_force": (b1, b2),
    }


def create_free_face_beam_elements(
    mesh_df,
    beam_youngs_modulus,
    beam_thickness,
    out_of_plane_width=1000.0,
    transformation_tag=1,
    beam_node_start_tag=None,
    beam_element_start_tag=None,
    coordinate_tolerance=1.0e-6,
):
    """Create a bonded beam along x=0, y=0, z>=0."""
    if beam_youngs_modulus <= 0.0:
        raise ValueError("beam_youngs_modulus must be positive.")
    if beam_thickness <= 0.0 or out_of_plane_width <= 0.0:
        raise ValueError("Beam dimensions must be positive.")

    coordinates = _node_coordinate_map(mesh_df)
    quads = _surface_quad_rows(mesh_df)

    edge_counts = Counter()
    for _, connectivity in quads:
        for index in range(4):
            edge = tuple(
                sorted((connectivity[index], connectivity[(index + 1) % 4]))
            )
            edge_counts[edge] += 1

    def on_free_face(node_tag):
        x, y, z = coordinates[node_tag]
        return (
            abs(x) <= coordinate_tolerance
            and abs(y) <= coordinate_tolerance
            and z >= -coordinate_tolerance
        )

    free_face_edges = []
    for edge, count in edge_counts.items():
        if count != 1 or not all(on_free_face(tag) for tag in edge):
            continue
        node_i, node_j = edge
        z_i = coordinates[node_i][2]
        z_j = coordinates[node_j][2]
        if np.isclose(z_i, z_j, atol=coordinate_tolerance, rtol=0.0):
            continue
        free_face_edges.append(
            (node_i, node_j) if z_i < z_j else (node_j, node_i)
        )

    free_face_edges.sort(key=lambda edge: coordinates[edge[0]][2])
    if not free_face_edges:
        raise ValueError("No exterior edges were found on the positive Z-axis.")

    soil_node_tags = sorted(
        {tag for edge in free_face_edges for tag in edge},
        key=lambda tag: coordinates[tag][2],
    )

    bottom = coordinates[soil_node_tags[0]]
    if not all(abs(value) <= coordinate_tolerance for value in bottom):
        raise ValueError(
            f"The detected free face starts at {bottom}, not (0, 0, 0)."
        )

    expected_edges = {
        tuple(sorted((soil_node_tags[index], soil_node_tags[index + 1])))
        for index in range(len(soil_node_tags) - 1)
    }
    if expected_edges != {tuple(sorted(edge)) for edge in free_face_edges}:
        raise ValueError("The free-face edges are not one continuous chain.")

    existing_nodes = {int(tag) for tag in ops.getNodeTags()}
    missing_nodes = [tag for tag in soil_node_tags if tag not in existing_nodes]
    if missing_nodes:
        raise ValueError(
            f"Free-face soil nodes are absent from OpenSees: {missing_nodes}"
        )

    if beam_node_start_tag is None:
        beam_node_start_tag = max(existing_nodes) + 1

    existing_elements = {int(tag) for tag in ops.getEleTags()}
    if beam_element_start_tag is None:
        beam_element_start_tag = max(existing_elements, default=0) + 1

    proposed_nodes = {
        beam_node_start_tag + index for index in range(len(soil_node_tags))
    }
    proposed_elements = {
        beam_element_start_tag + index
        for index in range(len(free_face_edges))
    }
    if proposed_nodes & existing_nodes:
        raise ValueError("The proposed beam node tags are not unique.")
    if proposed_elements & existing_elements:
        raise ValueError("The proposed beam element tags are not unique.")

    area = out_of_plane_width * beam_thickness
    second_moment = out_of_plane_width * beam_thickness**3 / 12.0

    # Existing SSPquad nodes retain two DOFs. New beam nodes have three.
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    soil_to_beam = {}
    for index, soil_node_tag in enumerate(soil_node_tags):
        beam_node_tag = beam_node_start_tag + index
        x, _, z = coordinates[soil_node_tag]
        ops.node(beam_node_tag, x, z)

        # The beam root and soil origin are both supported directly.
        # Do not place the fixed soil origin in an equalDOF constraint:
        # the Transformation handler does not safely follow a retained
        # node that is also fixed. All other beam nodes are bonded to the
        # soil surface in translation.
        is_origin = (
            abs(x) <= coordinate_tolerance
            and abs(z) <= coordinate_tolerance
        )
        if not is_origin:
            ops.equalDOF(soil_node_tag, beam_node_tag, 1, 2)

        soil_to_beam[soil_node_tag] = beam_node_tag

    ops.geomTransf("Linear", int(transformation_tag))

    beam_element_tags = []
    beam_connectivity = {}
    for index, (soil_i, soil_j) in enumerate(free_face_edges):
        element_tag = beam_element_start_tag + index
        beam_i = soil_to_beam[soil_i]
        beam_j = soil_to_beam[soil_j]

        ops.element(
            "elasticBeamColumn",
            element_tag,
            beam_i,
            beam_j,
            float(area),
            float(beam_youngs_modulus),
            float(second_moment),
            int(transformation_tag),
        )
        beam_element_tags.append(element_tag)
        beam_connectivity[element_tag] = (beam_i, beam_j)

    return {
        "soil_node_tags": soil_node_tags,
        "beam_node_tags": list(soil_to_beam.values()),
        "beam_element_tags": beam_element_tags,
        "soil_to_beam_node": soil_to_beam,
        "beam_connectivity": beam_connectivity,
        "cross_section_area": area,
        "second_moment_of_area": second_moment,
        "transformation_tag": int(transformation_tag),
    }


def apply_model_boundary_conditions(
    mesh_df,
    model_info,
    beam_info,
    coordinate_tolerance=1.0e-6,
):
    """Fix the positive X-axis soil base and the beam-root rotation."""
    coordinates = _node_coordinate_map(mesh_df)
    soil_node_tags = {int(tag) for tag in model_info["node_tags"]}

    support_nodes = sorted(
        (
            tag
            for tag in soil_node_tags
            if abs(coordinates[tag][1]) <= coordinate_tolerance
            and abs(coordinates[tag][2]) <= coordinate_tolerance
            and coordinates[tag][0] >= -coordinate_tolerance
        ),
        key=lambda tag: coordinates[tag][0],
    )
    if not support_nodes:
        raise ValueError("No soil support nodes were found on the positive X-axis.")

    origin_soil_nodes = [
        tag
        for tag in support_nodes
        if all(abs(value) <= coordinate_tolerance for value in coordinates[tag])
        and tag in beam_info["soil_to_beam_node"]
    ]
    if len(origin_soil_nodes) != 1:
        raise ValueError(
            "Expected exactly one beam-connected soil node at (0, 0, 0)."
        )

    origin_soil_node = origin_soil_nodes[0]
    origin_beam_node = int(
        beam_info["soil_to_beam_node"][origin_soil_node]
    )

    for node_tag in support_nodes:
        ops.fix(node_tag, 1, 1)

    # The beam root is not included in equalDOF. Fix all three beam DOFs
    # directly, as required for a fixed beam base.
    ops.fix(origin_beam_node, 1, 1, 1)

    return {
        "support_soil_node_tags": support_nodes,
        "origin_soil_node_tag": origin_soil_node,
        "origin_beam_node_tag": origin_beam_node,
        "soil_fixity": (1, 1),
        "beam_explicit_fixity": (1, 1, 1),
        "beam_effective_fixity": (1, 1, 1),
    }


def _convergence_diagnostics():
    """Return the current OpenSees convergence-test iteration information."""
    try:
        iterations = int(ops.testIter())
    except Exception:
        iterations = 0
    try:
        raw_norms = [float(value) for value in ops.testNorms()]
    except Exception:
        raw_norms = []

    used_count = min(max(iterations, 0), len(raw_norms))
    iteration_norms = raw_norms[:used_count]
    final_norm = iteration_norms[-1] if iteration_norms else math.nan
    return iterations, final_norm, tuple(iteration_norms)


def _maximum_soil_displacement(soil_node_tags):
    maximum = -1.0
    maximum_node = None
    for node_tag in soil_node_tags:
        ux = float(ops.nodeDisp(node_tag, 1))
        uz = float(ops.nodeDisp(node_tag, 2))
        magnitude = math.hypot(ux, uz)
        if magnitude > maximum:
            maximum = magnitude
            maximum_node = node_tag
    return maximum, maximum_node


def run_gravity_analysis(
    model_info,
    boundary_info,
    number_of_steps=100,
    tolerance=1.0e-5,
    maximum_iterations=100,
    time_series_tag=1,
    load_pattern_tag=1,
    hold_gravity_constant=True,
    progress_callback=None,
):
    """Apply SSPquad body forces incrementally and solve gravity."""
    if number_of_steps <= 0:
        raise ValueError("number_of_steps must be positive.")

    soil_element_tags = [int(tag) for tag in model_info["element_tags"]]
    soil_node_tags = [int(tag) for tag in model_info["node_tags"]]

    missing = set(soil_element_tags) - {int(tag) for tag in ops.getEleTags()}
    if missing:
        raise ValueError(f"Soil elements are absent from OpenSees: {sorted(missing)}")

    ops.wipeAnalysis()
    ops.timeSeries("Linear", int(time_series_tag))
    ops.pattern("Plain", int(load_pattern_tag), int(time_series_tag))

    for start in range(0, len(soil_element_tags), 500):
        chunk = soil_element_tags[start:start + 500]
        ops.eleLoad(
            "-ele",
            *chunk,
            "-type",
            "-SelfWeight",
            1.0,
            1.0,
        )

    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("UmfPack")
    ops.test(
        "NormDispIncr",
        float(tolerance),
        int(maximum_iterations),
        0,
    )
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0 / number_of_steps)
    ops.analysis("Static")

    history = []
    for step in range(1, number_of_steps + 1):
        result = ops.analyze(1)
        iterations, final_norm, iteration_norms = _convergence_diagnostics()
        if result != 0:
            diagnostic = {
                "phase": "Gravity",
                "step": step,
                "total_steps": number_of_steps,
                "load_factor": float(ops.getTime()),
                "iterations": iterations,
                "convergence_norm": final_norm,
                "iteration_norms": iteration_norms,
                "tolerance": float(tolerance),
                "maximum_iterations": int(maximum_iterations),
                "converged": False,
                "return_code": int(result),
                "maximum_soil_displacement": math.nan,
                "maximum_displacement_node": None,
            }
            if progress_callback is not None:
                progress_callback(diagnostic)
            raise RuntimeError(
                "Gravity analysis failed to converge.\n"
                f"Step: {step} of {number_of_steps}\n"
                f"Committed load factor: {ops.getTime():.6g}\n"
                f"Convergence iterations: {iterations}\n"
                f"Final displacement-increment norm: {final_norm:.6g}\n"
                f"Requested tolerance: {tolerance:.6g}\n"
                f"OpenSees return code: {result}"
            )

        max_displacement, max_displacement_node = (
            _maximum_soil_displacement(soil_node_tags)
        )

        diagnostic = {
            "phase": "Gravity",
            "step": step,
            "total_steps": number_of_steps,
            "load_factor": float(ops.getTime()),
            "iterations": iterations,
            "convergence_norm": final_norm,
            "iteration_norms": iteration_norms,
            "tolerance": float(tolerance),
            "maximum_iterations": int(maximum_iterations),
            "converged": True,
            "return_code": 0,
            "maximum_soil_displacement": max_displacement,
            "maximum_displacement_node": max_displacement_node,
        }
        history.append(diagnostic)
        if progress_callback is not None:
            progress_callback(dict(diagnostic))

    ops.reactions()
    support_nodes = boundary_info["support_soil_node_tags"]
    soil_horizontal_reaction = sum(
        float(ops.nodeReaction(tag, 1)) for tag in support_nodes
    )
    soil_vertical_reaction = sum(
        float(ops.nodeReaction(tag, 2)) for tag in support_nodes
    )
    beam_root = boundary_info["origin_beam_node_tag"]
    beam_horizontal_reaction = float(ops.nodeReaction(beam_root, 1))
    beam_vertical_reaction = float(ops.nodeReaction(beam_root, 2))
    beam_base_moment = float(ops.nodeReaction(beam_root, 3))

    horizontal_reaction = (
        soil_horizontal_reaction + beam_horizontal_reaction
    )
    vertical_reaction = soil_vertical_reaction + beam_vertical_reaction
    final_load_factor = float(ops.getTime())

    if hold_gravity_constant:
        ops.loadConst("-time", 0.0)
        ops.wipeAnalysis()

    return {
        "converged": True,
        "number_of_steps": number_of_steps,
        "final_gravity_load_factor": final_load_factor,
        "gravity_held_constant": hold_gravity_constant,
        "horizontal_support_reaction": horizontal_reaction,
        "vertical_support_reaction": vertical_reaction,
        "soil_horizontal_support_reaction": soil_horizontal_reaction,
        "soil_vertical_support_reaction": soil_vertical_reaction,
        "beam_horizontal_support_reaction": beam_horizontal_reaction,
        "beam_vertical_support_reaction": beam_vertical_reaction,
        "beam_base_moment_reaction": beam_base_moment,
        "history": history,
    }


def apply_point_load(
    node_tag,
    magnitude,
    direction,
    model_info,
    beam_info,
    time_series_tag=2,
    load_pattern_tag=2,
    time_series_type="Linear",
):
    """Create a new load pattern and apply one nodal point load.

    Parameters
    ----------
    node_tag : int
        Existing OpenSees soil or beam node tag.

    magnitude : float
        Signed load magnitude. Translational loads use N and moments
        use N.mm in the current model unit system.

        Examples:
            +1000 with direction="x"  -> load to the right
            -1000 with direction="x"  -> load to the left
            +1000 with direction="z"  -> upward load
            -1000 with direction="z"  -> downward load

    direction : str or int
        Load direction. Accepted values are:

            "x", "horizontal" or 1
            "z", "vertical" or 2
            "moment", "rotation" or 3

        Moment loads are only valid on three-DOF beam nodes. A
        positive moment is counter-clockwise in the model X-Z plane.

    model_info : dict
        Dictionary returned by create_opensees_sspquad_model().

    beam_info : dict
        Dictionary returned by create_free_face_beam_elements().

    time_series_tag : int, default=2
        Unique time-series tag. Gravity uses tag 1 in this script.

    load_pattern_tag : int, default=2
        Unique load-pattern tag. Gravity uses tag 1 in this script.

    time_series_type : str, default="Linear"
        OpenSees time-series type for the new load pattern.

    Returns
    -------
    dict
        Information about the created point load and load pattern.

    Notes
    -----
    This function creates a new time series and load pattern. Use
    unique time-series and pattern tags for every separate call. To
    place several point loads in one pattern, use
    apply_point_loads() instead.
    """
    return apply_point_loads(
        point_loads=[(node_tag, magnitude, direction)],
        model_info=model_info,
        beam_info=beam_info,
        time_series_tag=time_series_tag,
        load_pattern_tag=load_pattern_tag,
        time_series_type=time_series_type,
    )


def apply_point_loads(
    point_loads,
    model_info,
    beam_info,
    time_series_tag=2,
    load_pattern_tag=2,
    time_series_type="Linear",
):
    """Apply one or more point loads within a single load pattern.

    Each entry in point_loads must have the form:

        (node_tag, signed_magnitude, direction)

    For example:

        [
            (beam_node, -10_000.0, "z"),
            (beam_node,  5_000.0, "x"),
            (beam_node,  2.0e6, "moment"),
        ]
    """
    if not point_loads:
        raise ValueError("point_loads cannot be empty.")

    soil_node_tags = {int(tag) for tag in model_info["node_tags"]}
    beam_node_tags = {int(tag) for tag in beam_info["beam_node_tags"]}
    existing_node_tags = {int(tag) for tag in ops.getNodeTags()}

    direction_aliases = {
        "x": 1,
        "horizontal": 1,
        "fx": 1,
        "1": 1,
        "z": 2,
        "vertical": 2,
        "fz": 2,
        "2": 2,
        "moment": 3,
        "rotation": 3,
        "m": 3,
        "3": 3,
    }

    prepared_loads = []

    for load_definition in point_loads:
        if len(load_definition) != 3:
            raise ValueError(
                "Each point load must be "
                "(node_tag, magnitude, direction)."
            )

        node_tag, magnitude, direction = load_definition
        node_tag = int(node_tag)
        magnitude = float(magnitude)

        if not math.isfinite(magnitude):
            raise ValueError(
                f"Load magnitude at node {node_tag} is not finite."
            )

        if node_tag not in existing_node_tags:
            raise ValueError(
                f"Node {node_tag} does not exist in the OpenSees model."
            )

        if isinstance(direction, str):
            direction_key = direction.strip().lower()
        elif isinstance(direction, (int, np.integer)):
            direction_key = str(int(direction))
        else:
            raise TypeError(
                "direction must be a string or an integer DOF number."
            )

        if direction_key not in direction_aliases:
            raise ValueError(
                f"Unsupported direction {direction!r}. Use 'x', 'z' "
                "or 'moment' (equivalently DOF 1, 2 or 3)."
            )

        dof = direction_aliases[direction_key]

        if node_tag in beam_node_tags:
            node_type = "beam"
            load_vector = [0.0, 0.0, 0.0]
        elif node_tag in soil_node_tags:
            node_type = "soil"
            load_vector = [0.0, 0.0]
        else:
            raise ValueError(
                f"Node {node_tag} exists but is not listed as a soil "
                "or beam node in model_info/beam_info."
            )

        if dof == 3 and node_type != "beam":
            raise ValueError(
                f"A moment cannot be applied to soil node {node_tag}; "
                "SSPquad nodes have only two translational DOFs."
            )

        load_vector[dof - 1] = magnitude
        prepared_loads.append(
            {
                "node_tag": node_tag,
                "node_type": node_type,
                "direction_dof": dof,
                "magnitude": magnitude,
                "load_vector": tuple(load_vector),
            }
        )

    ops.timeSeries(time_series_type, int(time_series_tag))
    ops.pattern(
        "Plain",
        int(load_pattern_tag),
        int(time_series_tag),
    )

    for load_data in prepared_loads:
        ops.load(
            load_data["node_tag"],
            *load_data["load_vector"],
        )

    return {
        "time_series_tag": int(time_series_tag),
        "load_pattern_tag": int(load_pattern_tag),
        "time_series_type": time_series_type,
        "loads": prepared_loads,
    }


def _exterior_soil_path(mesh_df, start_node, end_node):
    """Return the shortest connected exterior-edge path between two nodes."""
    start_node = int(start_node)
    end_node = int(end_node)
    coordinates = _node_coordinate_map(mesh_df)
    edge_counts = Counter()

    for _, connectivity in _surface_quad_rows(mesh_df):
        for index, node_i in enumerate(connectivity):
            node_j = connectivity[(index + 1) % len(connectivity)]
            edge_counts[tuple(sorted((int(node_i), int(node_j))))] += 1

    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    boundary_nodes = {tag for edge in boundary_edges for tag in edge}
    for tag, label in ((start_node, "start"), (end_node, "end")):
        if tag not in boundary_nodes:
            raise ValueError(
                f"Soil line-load {label} node {tag} is not on the exterior "
                "boundary of the SSPquad mesh."
            )

    adjacency = {tag: [] for tag in boundary_nodes}
    for node_i, node_j in boundary_edges:
        x_i, _, z_i = coordinates[node_i]
        x_j, _, z_j = coordinates[node_j]
        length = math.hypot(x_j - x_i, z_j - z_i)
        if length <= 0.0:
            continue
        adjacency[node_i].append((node_j, length))
        adjacency[node_j].append((node_i, length))

    distances = {start_node: 0.0}
    previous = {}
    queue = [(0.0, start_node)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance > distances.get(node, math.inf):
            continue
        if node == end_node:
            break
        for neighbour, edge_length in adjacency[node]:
            candidate = distance + edge_length
            if candidate < distances.get(neighbour, math.inf):
                distances[neighbour] = candidate
                previous[neighbour] = node
                heapq.heappush(queue, (candidate, neighbour))

    if end_node not in distances:
        raise ValueError(
            f"No connected exterior soil path exists between nodes "
            f"{start_node} and {end_node}."
        )

    path = [end_node]
    while path[-1] != start_node:
        path.append(previous[path[-1]])
    path.reverse()
    return path


def _beam_path(beam_info, start_node, end_node):
    """Return beam nodes and elements along the unique free-face chain."""
    start_node = int(start_node)
    end_node = int(end_node)
    beam_nodes = {int(tag) for tag in beam_info["beam_node_tags"]}
    for tag, label in ((start_node, "start"), (end_node, "end")):
        if tag not in beam_nodes:
            raise ValueError(
                f"Beam line-load {label} node {tag} is not a free-face "
                "beam node."
            )

    adjacency = {tag: [] for tag in beam_nodes}
    for element_tag, (node_i, node_j) in beam_info["beam_connectivity"].items():
        element_tag = int(element_tag)
        node_i = int(node_i)
        node_j = int(node_j)
        adjacency[node_i].append((node_j, element_tag))
        adjacency[node_j].append((node_i, element_tag))

    previous = {start_node: (None, None)}
    queue = [start_node]
    while queue:
        node = queue.pop(0)
        if node == end_node:
            break
        for neighbour, element_tag in adjacency[node]:
            if neighbour not in previous:
                previous[neighbour] = (node, element_tag)
                queue.append(neighbour)

    if end_node not in previous:
        raise ValueError(
            f"No connected beam path exists between nodes "
            f"{start_node} and {end_node}."
        )

    nodes = [end_node]
    elements = []
    while nodes[-1] != start_node:
        parent, element_tag = previous[nodes[-1]]
        elements.append(element_tag)
        nodes.append(parent)
    nodes.reverse()
    elements.reverse()
    return nodes, elements


def apply_line_loads(
    line_loads,
    mesh_df,
    model_info,
    beam_info,
    time_series_tag=3,
    load_pattern_tag=3,
    time_series_type="Linear",
):
    """Apply uniform or trapezoidal boundary loads in global X or Z.

    Each load is a dictionary with ``target`` (``"Soil"`` or ``"Beam"``),
    ``start_node``, ``end_node``, ``start_magnitude``, ``end_magnitude`` and
    ``direction`` (``"x"`` or ``"z"``). Intensities use force per length.

    Soil boundary tractions are converted to consistent nodal forces on each
    exterior SSPquad edge. Beam loads are applied with ``eleLoad`` to every
    elasticBeamColumn on the selected free-face segment. A trapezoidal beam
    load is represented by the exact average intensity on each member.
    """
    if not line_loads:
        raise ValueError("line_loads cannot be empty.")

    coordinates = _node_coordinate_map(mesh_df)
    beam_to_soil = {
        int(beam_node): int(soil_node)
        for soil_node, beam_node in beam_info["soil_to_beam_node"].items()
    }
    prepared = []

    for index, definition in enumerate(line_loads, start=1):
        required = {
            "target",
            "start_node",
            "end_node",
            "start_magnitude",
            "end_magnitude",
            "direction",
        }
        missing = required - set(definition)
        if missing:
            raise ValueError(
                f"Line load {index} is missing: {', '.join(sorted(missing))}."
            )

        target = str(definition["target"]).strip().lower()
        if target not in {"soil", "beam"}:
            raise ValueError(f"Line load {index}: target must be Soil or Beam.")
        direction = str(definition["direction"]).strip().lower()
        if direction not in {"x", "z"}:
            raise ValueError(f"Line load {index}: direction must be x or z.")

        start_node = int(definition["start_node"])
        end_node = int(definition["end_node"])
        q_start = float(definition["start_magnitude"])
        q_end = float(definition["end_magnitude"])
        if start_node == end_node:
            raise ValueError(f"Line load {index}: start and end nodes must differ.")
        if not all(math.isfinite(value) for value in (q_start, q_end)):
            raise ValueError(f"Line load {index}: intensities must be finite.")

        if target == "soil":
            path_nodes = _exterior_soil_path(mesh_df, start_node, end_node)
            path_elements = []
            path_coordinates = [coordinates[tag] for tag in path_nodes]
        else:
            path_nodes, path_elements = _beam_path(
                beam_info,
                start_node,
                end_node,
            )
            path_coordinates = [coordinates[beam_to_soil[tag]] for tag in path_nodes]

        edge_lengths = [
            math.hypot(
                path_coordinates[i + 1][0] - path_coordinates[i][0],
                path_coordinates[i + 1][2] - path_coordinates[i][2],
            )
            for i in range(len(path_nodes) - 1)
        ]
        total_length = sum(edge_lengths)
        if total_length <= 0.0:
            raise ValueError(f"Line load {index}: selected path has zero length.")

        stations = [0.0]
        for length in edge_lengths:
            stations.append(stations[-1] + length)
        intensities = [
            q_start + (q_end - q_start) * station / total_length
            for station in stations
        ]
        prepared.append(
            {
                "target": target,
                "start_node": start_node,
                "end_node": end_node,
                "direction": direction,
                "start_magnitude": q_start,
                "end_magnitude": q_end,
                "path_nodes": path_nodes,
                "path_elements": path_elements,
                "edge_lengths": edge_lengths,
                "stations": stations,
                "intensities": intensities,
                "total_length": total_length,
                "resultant": 0.5 * (q_start + q_end) * total_length,
            }
        )

    ops.timeSeries(time_series_type, int(time_series_tag))
    ops.pattern("Plain", int(load_pattern_tag), int(time_series_tag))

    soil_nodal_forces = {}
    for load_data in prepared:
        direction_index = 0 if load_data["direction"] == "x" else 1
        if load_data["target"] == "soil":
            for edge_index, length in enumerate(load_data["edge_lengths"]):
                node_i = load_data["path_nodes"][edge_index]
                node_j = load_data["path_nodes"][edge_index + 1]
                q_i = load_data["intensities"][edge_index]
                q_j = load_data["intensities"][edge_index + 1]
                force_i = length * (2.0 * q_i + q_j) / 6.0
                force_j = length * (q_i + 2.0 * q_j) / 6.0
                soil_nodal_forces.setdefault(node_i, [0.0, 0.0])[direction_index] += force_i
                soil_nodal_forces.setdefault(node_j, [0.0, 0.0])[direction_index] += force_j
        else:
            for edge_index, element_tag in enumerate(load_data["path_elements"]):
                q_average = 0.5 * (
                    load_data["intensities"][edge_index]
                    + load_data["intensities"][edge_index + 1]
                )
                # A vertical member's local x is global +Z and local y is
                # global -X. OpenSees 2D beamUniform arguments are Wy, Wx.
                wy = -q_average if load_data["direction"] == "x" else 0.0
                wx = q_average if load_data["direction"] == "z" else 0.0
                ops.eleLoad(
                    "-ele",
                    int(element_tag),
                    "-type",
                    "-beamUniform",
                    float(wy),
                    float(wx),
                )

    for node_tag, force_vector in soil_nodal_forces.items():
        ops.load(int(node_tag), *map(float, force_vector))

    return {
        "time_series_tag": int(time_series_tag),
        "load_pattern_tag": int(load_pattern_tag),
        "time_series_type": time_series_type,
        "loads": prepared,
        "soil_nodal_forces": {
            int(tag): tuple(values) for tag, values in soil_nodal_forces.items()
        },
    }


def run_point_load_analysis(
    model_info,
    beam_info,
    boundary_info,
    number_of_steps=100,
    tolerance=1.0e-5,
    maximum_iterations=100,
    hold_point_loads_constant=False,
    progress_callback=None,
):
    """Run the post-gravity static analysis for all active load patterns."""
    if number_of_steps <= 0:
        raise ValueError("number_of_steps must be positive.")

    soil_node_tags = [int(tag) for tag in model_info["node_tags"]]
    beam_element_tags = [int(tag) for tag in beam_info["beam_element_tags"]]

    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("UmfPack")
    ops.test(
        "NormDispIncr",
        float(tolerance),
        int(maximum_iterations),
        0,
    )
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0 / number_of_steps)
    ops.analysis("Static")

    history = []
    for step in range(1, number_of_steps + 1):
        result = ops.analyze(1)
        iterations, final_norm, iteration_norms = _convergence_diagnostics()
        if result != 0:
            diagnostic = {
                "phase": "Applied loads",
                "step": step,
                "total_steps": number_of_steps,
                "load_factor": float(ops.getTime()),
                "iterations": iterations,
                "convergence_norm": final_norm,
                "iteration_norms": iteration_norms,
                "tolerance": float(tolerance),
                "maximum_iterations": int(maximum_iterations),
                "converged": False,
                "return_code": int(result),
                "maximum_soil_displacement": math.nan,
                "maximum_displacement_node": None,
            }
            if progress_callback is not None:
                progress_callback(diagnostic)
            raise RuntimeError(
                "Post-gravity static analysis failed to converge.\n"
                f"Step: {step} of {number_of_steps}\n"
                f"Committed load factor: {ops.getTime():.6g}\n"
                f"Convergence iterations: {iterations}\n"
                f"Final displacement-increment norm: {final_norm:.6g}\n"
                f"Requested tolerance: {tolerance:.6g}\n"
                f"OpenSees return code: {result}"
            )

        max_displacement, max_displacement_node = (
            _maximum_soil_displacement(soil_node_tags)
        )

        diagnostic = {
            "phase": "Applied loads",
            "step": step,
            "total_steps": number_of_steps,
            "load_factor": float(ops.getTime()),
            "iterations": iterations,
            "convergence_norm": final_norm,
            "iteration_norms": iteration_norms,
            "tolerance": float(tolerance),
            "maximum_iterations": int(maximum_iterations),
            "converged": True,
            "return_code": 0,
            "maximum_soil_displacement": max_displacement,
            "maximum_displacement_node": max_displacement_node,
        }
        history.append(diagnostic)
        if progress_callback is not None:
            progress_callback(dict(diagnostic))

    ops.reactions()
    support_nodes = boundary_info["support_soil_node_tags"]
    beam_root = boundary_info["origin_beam_node_tag"]

    soil_horizontal_reaction = sum(
        float(ops.nodeReaction(tag, 1)) for tag in support_nodes
    )
    soil_vertical_reaction = sum(
        float(ops.nodeReaction(tag, 2)) for tag in support_nodes
    )
    beam_horizontal_reaction = float(ops.nodeReaction(beam_root, 1))
    beam_vertical_reaction = float(ops.nodeReaction(beam_root, 2))
    beam_base_moment = float(ops.nodeReaction(beam_root, 3))

    beam_local_forces = {
        element_tag: tuple(
            float(value)
            for value in ops.eleResponse(element_tag, "localForce")
        )
        for element_tag in beam_element_tags
    }

    final_load_factor = float(ops.getTime())

    if hold_point_loads_constant:
        ops.loadConst("-time", 0.0)
        ops.wipeAnalysis()

    return {
        "converged": True,
        "number_of_steps": number_of_steps,
        "final_point_load_factor": final_load_factor,
        "final_load_factor": final_load_factor,
        "point_loads_held_constant": hold_point_loads_constant,
        "horizontal_support_reaction": (
            soil_horizontal_reaction + beam_horizontal_reaction
        ),
        "vertical_support_reaction": (
            soil_vertical_reaction + beam_vertical_reaction
        ),
        "beam_base_moment_reaction": beam_base_moment,
        "beam_local_forces": beam_local_forces,
        "history": history,
    }


def get_beam_local_forces(beam_info):
    """Return current local end forces for every free-face beam element."""
    return {
        int(element_tag): tuple(
            float(value)
            for value in ops.eleResponse(int(element_tag), "localForce")
        )
        for element_tag in beam_info["beam_element_tags"]
    }


def run_complete_analysis_case(
    mesh_df,
    case_name,
    material_model,
    youngs_modulus,
    poissons_ratio,
    density,
    friction_angle,
    cohesion,
    dilation_angle,
    thickness,
    body_force,
    atmospheric_pressure,
    hardening_ratio,
    beam_youngs_modulus,
    beam_thickness,
    beam_width,
    point_loads=None,
    line_loads=None,
    number_of_gravity_steps=100,
    number_of_point_load_steps=100,
    point_load_time_series_tag=2,
    point_load_pattern_tag=2,
    line_load_time_series_tag=3,
    line_load_pattern_tag=3,
    coordinate_tolerance=1.0e-6,
    tolerance=1.0e-5,
    maximum_iterations=100,
    progress_callback=None,
):
    """Build and solve one complete Drucker-Prager analysis case."""
    model_info = create_opensees_sspquad_model(
        mesh_df=mesh_df,
        youngs_modulus=youngs_modulus,
        poissons_ratio=poissons_ratio,
        density=density,
        friction_angle=friction_angle,
        cohesion=cohesion,
        dilation_angle=dilation_angle,
        thickness=thickness,
        material_tag=1,
        coordinate_columns=("x", "z"),
        body_force=body_force,
        atmospheric_pressure=atmospheric_pressure,
        hardening_ratio=hardening_ratio,
        material_model=material_model,
    )

    beam_info = create_free_face_beam_elements(
        mesh_df=mesh_df,
        beam_youngs_modulus=beam_youngs_modulus,
        beam_thickness=beam_thickness,
        out_of_plane_width=beam_width,
        transformation_tag=1,
        coordinate_tolerance=coordinate_tolerance,
    )

    boundary_info = apply_model_boundary_conditions(
        mesh_df=mesh_df,
        model_info=model_info,
        beam_info=beam_info,
        coordinate_tolerance=coordinate_tolerance,
    )

    def case_progress(diagnostic):
        if progress_callback is not None:
            enriched = dict(diagnostic)
            enriched["case_name"] = str(case_name)
            enriched["material_model"] = str(material_model)
            progress_callback(enriched)

    gravity_results = run_gravity_analysis(
        model_info=model_info,
        boundary_info=boundary_info,
        number_of_steps=number_of_gravity_steps,
        tolerance=tolerance,
        maximum_iterations=maximum_iterations,
        progress_callback=case_progress,
    )
    gravity_beam_local_forces = get_beam_local_forces(beam_info)

    point_load_info = None
    line_load_info = None
    point_results = None
    if point_loads:
        point_load_info = apply_point_loads(
            point_loads=point_loads,
            model_info=model_info,
            beam_info=beam_info,
            time_series_tag=point_load_time_series_tag,
            load_pattern_tag=point_load_pattern_tag,
        )
    if line_loads:
        line_load_info = apply_line_loads(
            line_loads=line_loads,
            mesh_df=mesh_df,
            model_info=model_info,
            beam_info=beam_info,
            time_series_tag=line_load_time_series_tag,
            load_pattern_tag=line_load_pattern_tag,
        )

    if point_loads or line_loads:
        point_results = run_point_load_analysis(
            model_info=model_info,
            beam_info=beam_info,
            boundary_info=boundary_info,
            number_of_steps=number_of_point_load_steps,
            tolerance=tolerance,
            maximum_iterations=maximum_iterations,
            progress_callback=case_progress,
        )
        final_beam_local_forces = point_results["beam_local_forces"]
        final_results = point_results
    else:
        final_beam_local_forces = gravity_beam_local_forces
        final_results = gravity_results

    expected_soil_weight = (
        model_info["total_planar_area"]
        * model_info["thickness"]
        * abs(model_info["body_force"][1])
    )

    return {
        "case_name": str(case_name),
        "material_model": model_info["material_model"],
        "model_info": model_info,
        "beam_info": beam_info,
        "boundary_info": boundary_info,
        "gravity_results": gravity_results,
        "gravity_beam_local_forces": gravity_beam_local_forces,
        "point_load_info": point_load_info,
        "line_load_info": line_load_info,
        "point_results": point_results,
        "final_results": final_results,
        "final_beam_local_forces": final_beam_local_forces,
        "expected_soil_weight": expected_soil_weight,
    }


def _maximum_beam_actions(beam_local_forces):
    maximum_shear = 0.0
    maximum_moment = 0.0

    for forces in beam_local_forces.values():
        maximum_shear = max(
            maximum_shear,
            abs(float(forces[1])),
            abs(float(forces[4])),
        )
        maximum_moment = max(
            maximum_moment,
            abs(float(forces[2])),
            abs(float(forces[5])),
        )

    return maximum_shear, maximum_moment


def print_analysis_case_results(case):
    """Print gravity and final-load results for one material case."""
    gravity_results = case["gravity_results"]
    gravity_step = gravity_results["history"][-1]
    expected_weight = case["expected_soil_weight"]
    vertical_reaction = abs(gravity_results["vertical_support_reaction"])
    equilibrium_error = vertical_reaction - expected_weight
    relative_error = (
        equilibrium_error / expected_weight
        if expected_weight != 0.0
        else math.nan
    )

    print(f"\n{case['case_name']} - gravity results")
    print("-" * (len(case["case_name"]) + 18))
    print(f"Converged: {gravity_results['converged']}")
    print(
        "Maximum soil displacement: "
        f"{gravity_step['maximum_soil_displacement']:.6g} mm"
    )
    print(
        "Maximum-displacement node: "
        f"{gravity_step['maximum_displacement_node']}"
    )
    print(
        "Horizontal support reaction: "
        f"{gravity_results['horizontal_support_reaction']:.6g} N"
    )
    print(
        "Vertical support reaction: "
        f"{gravity_results['vertical_support_reaction']:.6g} N"
    )
    print(
        "Beam base moment: "
        f"{gravity_results['beam_base_moment_reaction']:.6g} N.mm"
    )
    print(f"Expected soil weight: {expected_weight:.6g} N")
    print(f"Gravity force-balance error: {equilibrium_error:.6g} N")
    print(
        "Gravity relative force-balance error: "
        f"{100.0 * relative_error:.6g} %"
    )

    point_results = case["point_results"]
    if point_results is not None:
        final_step = point_results["history"][-1]
        maximum_shear, maximum_moment = _maximum_beam_actions(
            case["final_beam_local_forces"]
        )
        print(f"\n{case['case_name']} - gravity plus point loads")
        print("-" * (len(case["case_name"]) + 29))
        print(f"Converged: {point_results['converged']}")
        print(
            "Maximum soil displacement: "
            f"{final_step['maximum_soil_displacement']:.6g} mm"
        )
        print(
            "Maximum-displacement node: "
            f"{final_step['maximum_displacement_node']}"
        )
        print(
            "Horizontal support reaction: "
            f"{point_results['horizontal_support_reaction']:.6g} N"
        )
        print(
            "Vertical support reaction: "
            f"{point_results['vertical_support_reaction']:.6g} N"
        )
        print(
            "Beam base moment: "
            f"{point_results['beam_base_moment_reaction']:.6g} N.mm"
        )
        print(f"Maximum beam shear magnitude: {maximum_shear:.6g} N")
        print(f"Maximum beam moment magnitude: {maximum_moment:.6g} N.mm")


def plot_model_and_beam_forces(
    mesh_df,
    model_info,
    beam_info,
    point_load_info,
    point_results,
    show_node_tags=True,
    save_path=None,
    show=True,
):
    """Plot the model, shear-force diagram and bending-moment diagram.

    Beam local-force output is interpreted as OpenSees end actions:

        (Ni, Vi, Mi, Nj, Vj, Mj)

    The J-end shear and moment signs are reversed when drawing member
    section diagrams. Shear is displayed in kN and moment in kN.m.
    """
    coordinates = _node_coordinate_map(mesh_df)
    beam_forces = point_results["beam_local_forces"]

    figure, (axis_model, axis_shear, axis_moment) = plt.subplots(
        1,
        3,
        figsize=(16, 8),
        gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]},
        sharey=True,
    )

    # ------------------------------------------------------
    # Undeformed soil mesh and beam
    # ------------------------------------------------------

    for connectivity in model_info["element_connectivity"].values():
        points = [
            (coordinates[node_tag][0], coordinates[node_tag][2])
            for node_tag in connectivity
        ]
        points.append(points[0])
        x_values, z_values = zip(*points)

        axis_model.fill(
            x_values,
            z_values,
            facecolor="#eadfca",
            edgecolor="#8b8174",
            linewidth=0.65,
            alpha=0.72,
        )

    soil_face_nodes = beam_info["soil_node_tags"]
    face_x = [coordinates[tag][0] for tag in soil_face_nodes]
    face_z = [coordinates[tag][2] for tag in soil_face_nodes]
    axis_model.plot(
        face_x,
        face_z,
        color="#25364a",
        linewidth=4.0,
        label="Beam/facing",
        zorder=5,
    )

    axis_model.scatter(
        face_x,
        face_z,
        color="#25364a",
        s=28,
        zorder=6,
    )

    if show_node_tags:
        for soil_node, beam_node in beam_info["soil_to_beam_node"].items():
            x, _, z = coordinates[soil_node]
            axis_model.annotate(
                f"S{soil_node}/B{beam_node}",
                (x, z),
                xytext=(7, 3),
                textcoords="offset points",
                fontsize=7,
                color="#25364a",
            )

    # Draw point-load arrows and labels.
    all_x = [value[0] for value in coordinates.values()]
    all_z = [value[2] for value in coordinates.values()]
    x_span = max(all_x) - min(all_x)
    z_span = max(all_z) - min(all_z)
    arrow_length = 0.12 * max(x_span, z_span)

    for load_data in point_load_info["loads"]:
        node_tag = load_data["node_tag"]
        dof = load_data["direction_dof"]
        magnitude = load_data["magnitude"]

        if node_tag in coordinates:
            x, _, z = coordinates[node_tag]
        else:
            x, z = map(float, ops.nodeCoord(node_tag)[:2])

        if dof in (1, 2) and magnitude != 0.0:
            sign = 1.0 if magnitude > 0.0 else -1.0
            dx = sign * arrow_length if dof == 1 else 0.0
            dz = sign * arrow_length if dof == 2 else 0.0
            start = (x - dx, z - dz)

            axis_model.annotate(
                "",
                xy=(x, z),
                xytext=start,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "#c53b32",
                    "linewidth": 2.0,
                },
                zorder=8,
            )
            axis_model.annotate(
                f"{magnitude / 1000.0:.3g} kN",
                start,
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                color="#a52f28",
            )
        elif dof == 3 and magnitude != 0.0:
            radius = 0.07 * max(x_span, z_span)
            if magnitude > 0.0:
                start = (x + radius, z)
                end = (x, z + radius)
                connection = "arc3,rad=0.55"
            else:
                start = (x, z + radius)
                end = (x + radius, z)
                connection = "arc3,rad=-0.55"

            moment_arrow = FancyArrowPatch(
                start,
                end,
                connectionstyle=connection,
                arrowstyle="-|>",
                mutation_scale=12,
                color="#8c2d7a",
                linewidth=2.0,
                zorder=8,
            )
            axis_model.add_patch(moment_arrow)
            axis_model.annotate(
                f"{magnitude / 1.0e6:.3g} kN.m",
                (x + radius, z + radius),
                fontsize=8,
                color="#8c2d7a",
            )

    axis_model.set_title("Soil mesh, beam and point loads")
    axis_model.set_xlabel("Global X (mm)")
    axis_model.set_ylabel("Global Z (mm)")
    axis_model.set_aspect("equal", adjustable="box")
    axis_model.grid(True, alpha=0.22)
    axis_model.legend(loc="best")

    # ------------------------------------------------------
    # Beam force diagrams
    # ------------------------------------------------------

    shear_limits = [0.0]
    moment_limits = [0.0]

    ordered_elements = sorted(
        beam_info["beam_element_tags"],
        key=lambda tag: min(
            ops.nodeCoord(beam_info["beam_connectivity"][tag][0])[1],
            ops.nodeCoord(beam_info["beam_connectivity"][tag][1])[1],
        ),
    )

    for element_tag in ordered_elements:
        node_i, node_j = beam_info["beam_connectivity"][element_tag]
        _, z_i = map(float, ops.nodeCoord(node_i)[:2])
        _, z_j = map(float, ops.nodeCoord(node_j)[:2])
        forces = beam_forces[element_tag]

        # Convert nodal end actions to consistent internal section signs.
        shear_i = forces[1] / 1000.0
        shear_j = -forces[4] / 1000.0
        moment_i = forces[2] / 1.0e6
        moment_j = -forces[5] / 1.0e6

        shear_limits.extend((shear_i, shear_j))
        moment_limits.extend((moment_i, moment_j))

        axis_shear.plot(
            (shear_i, shear_j),
            (z_i, z_j),
            color="#187a72",
            linewidth=2.2,
        )
        axis_shear.fill_betweenx(
            (z_i, z_j),
            (0.0, 0.0),
            (shear_i, shear_j),
            color="#4db6ac",
            alpha=0.32,
        )

        axis_moment.plot(
            (moment_i, moment_j),
            (z_i, z_j),
            color="#7756a5",
            linewidth=2.2,
        )
        axis_moment.fill_betweenx(
            (z_i, z_j),
            (0.0, 0.0),
            (moment_i, moment_j),
            color="#9b7bc4",
            alpha=0.32,
        )

        axis_shear.scatter(
            (shear_i, shear_j),
            (z_i, z_j),
            color="#187a72",
            s=14,
            zorder=4,
        )
        axis_moment.scatter(
            (moment_i, moment_j),
            (z_i, z_j),
            color="#7756a5",
            s=14,
            zorder=4,
        )

    for axis in (axis_shear, axis_moment):
        axis.axvline(0.0, color="#2d2d2d", linewidth=0.9)
        axis.axhline(0.0, color="#2d2d2d", linewidth=0.7)
        axis.grid(True, alpha=0.25)
        axis.set_ylabel("Global Z (mm)")

    shear_range = max(abs(value) for value in shear_limits)
    moment_range = max(abs(value) for value in moment_limits)
    if shear_range > 0.0:
        axis_shear.set_xlim(-1.15 * shear_range, 1.15 * shear_range)
    if moment_range > 0.0:
        axis_moment.set_xlim(-1.15 * moment_range, 1.15 * moment_range)

    axis_shear.set_title("Shear-force diagram")
    axis_shear.set_xlabel("Local shear V (kN)")
    axis_moment.set_title("Bending-moment diagram")
    axis_moment.set_xlabel("Local bending moment M (kN.m)")

    figure.suptitle(
        "Retaining-wall soil/beam response after gravity and point loads",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()

    if save_path is not None:
        figure.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Force-diagram plot saved to: {save_path}")

    if show:
        plt.show()

    return {
        "figure": figure,
        "model_axis": axis_model,
        "shear_axis": axis_shear,
        "moment_axis": axis_moment,
        "save_path": save_path,
    }


def main():
    soil_youngs_modulus = SOIL_E_KPA * KPA_TO_N_MM2
    soil_density = SOIL_DENSITY_MG_M3 * MG_M3_TO_N_S2_MM4
    atmospheric_pressure = (
        ATMOSPHERIC_PRESSURE_KPA * KPA_TO_N_MM2
    )
    soil_cohesion = SOIL_COHESION_KPA * KPA_TO_N_MM2
    gravity_acceleration = GRAVITY_M_S2 * M_TO_MM
    soil_unit_weight = soil_density * gravity_acceleration

    print("Unit system: N, mm, s")
    print(f"Soil E: {soil_youngs_modulus:g} N/mm^2")
    print(f"Soil density: {soil_density:g} N.s^2/mm^4")
    print(f"Soil unit weight: {soil_unit_weight:g} N/mm^3")

    if RECREATE_MESH:
        create_mesh(
            STP_LOCATION,
            MSH_LOCATION,
            show_gui=SHOW_GMSH_GUI,
        )

    mesh_df = read_msh_to_dataframe(MSH_LOCATION)

    common_case_arguments = {
        "mesh_df": mesh_df,
        "youngs_modulus": soil_youngs_modulus,
        "poissons_ratio": SOIL_POISSONS_RATIO,
        "density": soil_density,
        "friction_angle": SOIL_FRICTION_ANGLE_DEG,
        "cohesion": soil_cohesion,
        "dilation_angle": SOIL_DILATION_ANGLE_DEG,
        "thickness": MODEL_WIDTH_MM,
        "body_force": (0.0, -soil_unit_weight),
        "atmospheric_pressure": atmospheric_pressure,
        "hardening_ratio": SOIL_HARDENING_RATIO,
        "beam_youngs_modulus": BEAM_E_N_MM2,
        "beam_thickness": BEAM_THICKNESS_MM,
        "beam_width": BEAM_WIDTH_MM,
        "point_loads": POINT_LOADS,
        "number_of_gravity_steps": NUMBER_OF_GRAVITY_STEPS,
        "number_of_point_load_steps": NUMBER_OF_POINT_LOAD_STEPS,
        "point_load_time_series_tag": POINT_LOAD_TIME_SERIES_TAG,
        "point_load_pattern_tag": POINT_LOAD_PATTERN_TAG,
        "tolerance": 1.0e-5,
        "maximum_iterations": 100,
    }

    print("\nSolving Drucker-Prager model...")
    nonlinear_case = run_complete_analysis_case(
        case_name="Drucker-Prager",
        material_model="drucker_prager",
        **common_case_arguments,
    )
    print_analysis_case_results(nonlinear_case)

    print("\nDrucker-Prager material settings")
    print("--------------------------------")
    print(f"Hardening ratio H/G: {SOIL_HARDENING_RATIO:.6g}")
    print(
        "Hardening modulus H: "
        f"{nonlinear_case['model_info']['hardening_modulus']:.6g} N/mm^2"
    )

    beam_info = nonlinear_case["beam_info"]
    print("\nFree-face soil-to-beam node map")
    print("--------------------------------")
    for soil_node, beam_node in beam_info["soil_to_beam_node"].items():
        print(f"Soil node {soil_node} -> beam node {beam_node}")

    if POINT_LOADS:
        point_load_info = nonlinear_case["point_load_info"]
        point_results = nonlinear_case["point_results"]

        print("\nApplied point loads")
        print("-------------------")
        for load_data in point_load_info["loads"]:
            print(
                f"Node {load_data['node_tag']}: "
                f"vector {load_data['load_vector']}"
            )

        print("\nDrucker-Prager beam local end forces")
        print("-------------------------------------")
        print("Format: (Ni, Vi, Mi, Nj, Vj, Mj)")
        for element_tag, forces in point_results["beam_local_forces"].items():
            print(f"Beam {element_tag}: {forces}")

        plot_path = RESULTS_PLOT_PATH if SAVE_RESULTS_PLOT else None
        plot_model_and_beam_forces(
            mesh_df=mesh_df,
            model_info=nonlinear_case["model_info"],
            beam_info=beam_info,
            point_load_info=point_load_info,
            point_results=point_results,
            show_node_tags=SHOW_NODE_TAGS_ON_PLOT,
            save_path=plot_path,
            show=SHOW_RESULTS_PLOT,
        )
    else:
        print("\nNo point loads requested: POINT_LOADS is empty.")

if __name__ == "__main__":
    main()
