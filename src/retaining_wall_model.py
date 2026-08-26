import ast
import math
from collections import Counter

import gmsh
import numpy as np
import openseespy.opensees as ops
import pandas as pd


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
    # (node_tag, signed_magnitude, direction)

    (7, -10_000.0, "z"),       # 10 kN downward
    (52, 5_000.0, "x"),        # 5 kN right
    (52, 2.0e6, "moment"),     # 2 kN.m counter-clockwise
]
NUMBER_OF_POINT_LOAD_STEPS = 100
POINT_LOAD_TIME_SERIES_TAG = 2
POINT_LOAD_PATTERN_TAG = 2

# Set this to False to reuse an existing .msh file.
RECREATE_MESH = True

# Set this to True if the Gmsh GUI should open after meshing.
SHOW_GMSH_GUI = False


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
    show_gui=False,
):
    """Create a quadrilateral surface mesh and write a Gmsh 2.2 file."""
    gmsh.initialize()

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
    gmsh.initialize()

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
):
    """Create a plane-strain Drucker-Prager/SSPquad soil model."""
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
    if not (0.0 <= dilation_angle <= friction_angle < 90.0):
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

    # The default remains elastic-perfectly plastic. A small positive
    # hardening_ratio may be used explicitly as numerical regularization.
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
        "material_type": "DruckerPrager",
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


def run_gravity_analysis(
    model_info,
    boundary_info,
    number_of_steps=100,
    tolerance=1.0e-5,
    maximum_iterations=100,
    time_series_tag=1,
    load_pattern_tag=1,
    hold_gravity_constant=True,
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
        if result != 0:
            raise RuntimeError(
                "Gravity analysis failed to converge.\n"
                f"Step: {step} of {number_of_steps}\n"
                f"Committed load factor: {ops.getTime():.6g}\n"
                f"OpenSees return code: {result}"
            )

        max_displacement = -1.0
        max_displacement_node = None
        for node_tag in soil_node_tags:
            ux = float(ops.nodeDisp(node_tag, 1))
            uy = float(ops.nodeDisp(node_tag, 2))
            magnitude = math.hypot(ux, uy)
            if magnitude > max_displacement:
                max_displacement = magnitude
                max_displacement_node = node_tag

        history.append(
            {
                "step": step,
                "load_factor": float(ops.getTime()),
                "maximum_soil_displacement": max_displacement,
                "maximum_displacement_node": max_displacement_node,
            }
        )

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


def run_point_load_analysis(
    model_info,
    beam_info,
    boundary_info,
    number_of_steps=100,
    tolerance=1.0e-5,
    maximum_iterations=100,
    hold_point_loads_constant=False,
):
    """Run the post-gravity static analysis for active point loads."""
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
        if result != 0:
            raise RuntimeError(
                "Point-load analysis failed to converge.\n"
                f"Step: {step} of {number_of_steps}\n"
                f"Committed load factor: {ops.getTime():.6g}\n"
                f"OpenSees return code: {result}"
            )

        max_displacement = -1.0
        max_displacement_node = None
        for node_tag in soil_node_tags:
            ux = float(ops.nodeDisp(node_tag, 1))
            uz = float(ops.nodeDisp(node_tag, 2))
            magnitude = math.hypot(ux, uz)
            if magnitude > max_displacement:
                max_displacement = magnitude
                max_displacement_node = node_tag

        history.append(
            {
                "step": step,
                "load_factor": float(ops.getTime()),
                "maximum_soil_displacement": max_displacement,
                "maximum_displacement_node": max_displacement_node,
            }
        )

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

    model_info = create_opensees_sspquad_model(
        mesh_df=mesh_df,
        youngs_modulus=soil_youngs_modulus,
        poissons_ratio=SOIL_POISSONS_RATIO,
        density=soil_density,
        friction_angle=SOIL_FRICTION_ANGLE_DEG,
        cohesion=soil_cohesion,
        dilation_angle=SOIL_DILATION_ANGLE_DEG,
        thickness=MODEL_WIDTH_MM,
        material_tag=1,
        coordinate_columns=("x", "z"),
        body_force=(0.0, -soil_unit_weight),
        atmospheric_pressure=atmospheric_pressure,
    )

    beam_info = create_free_face_beam_elements(
        mesh_df=mesh_df,
        beam_youngs_modulus=BEAM_E_N_MM2,
        beam_thickness=BEAM_THICKNESS_MM,
        out_of_plane_width=BEAM_WIDTH_MM,
        transformation_tag=1,
    )

    boundary_info = apply_model_boundary_conditions(
        mesh_df=mesh_df,
        model_info=model_info,
        beam_info=beam_info,
    )

    gravity_results = run_gravity_analysis(
        model_info=model_info,
        boundary_info=boundary_info,
        number_of_steps=NUMBER_OF_GRAVITY_STEPS,
        tolerance=1.0e-5,
        maximum_iterations=100,
    )

    expected_soil_weight = (
        model_info["total_planar_area"]
        * model_info["thickness"]
        * abs(model_info["body_force"][1])
    )
    vertical_reaction = abs(gravity_results["vertical_support_reaction"])
    force_balance_error = vertical_reaction - expected_soil_weight
    relative_error = (
        force_balance_error / expected_soil_weight
        if expected_soil_weight != 0.0
        else math.nan
    )
    final_step = gravity_results["history"][-1]

    print("\nGravity analysis results")
    print("------------------------")
    print(f"Converged: {gravity_results['converged']}")
    print(
        "Final gravity load factor: "
        f"{gravity_results['final_gravity_load_factor']:.6g}"
    )
    print(
        "Total horizontal support reaction: "
        f"{gravity_results['horizontal_support_reaction']:.6g} N"
    )
    print(
        "Total vertical support reaction: "
        f"{gravity_results['vertical_support_reaction']:.6g} N"
    )
    print(
        "  Soil-base vertical reaction: "
        f"{gravity_results['soil_vertical_support_reaction']:.6g} N"
    )
    print(
        "  Beam-root vertical reaction: "
        f"{gravity_results['beam_vertical_support_reaction']:.6g} N"
    )
    print(
        "Beam base moment: "
        f"{gravity_results['beam_base_moment_reaction']:.6g} N.mm"
    )
    print(
        "Maximum soil displacement: "
        f"{final_step['maximum_soil_displacement']:.6g} mm"
    )
    print(
        "Maximum-displacement node: "
        f"{final_step['maximum_displacement_node']}"
    )

    print("\nGravity equilibrium check")
    print("-------------------------")
    print(f"Expected soil weight: {expected_soil_weight:.6g} N")
    print(f"Vertical reaction magnitude: {vertical_reaction:.6g} N")
    print(f"Force-balance error: {force_balance_error:.6g} N")
    print(f"Relative force-balance error: {100.0 * relative_error:.6g} %")

    print("\nFree-face soil-to-beam node map")
    print("--------------------------------")
    for soil_node, beam_node in beam_info["soil_to_beam_node"].items():
        print(f"Soil node {soil_node} -> beam node {beam_node}")

    if POINT_LOADS:
        point_load_info = apply_point_loads(
            point_loads=POINT_LOADS,
            model_info=model_info,
            beam_info=beam_info,
            time_series_tag=POINT_LOAD_TIME_SERIES_TAG,
            load_pattern_tag=POINT_LOAD_PATTERN_TAG,
        )

        print("\nApplied point loads")
        print("-------------------")
        for load_data in point_load_info["loads"]:
            print(
                f"Node {load_data['node_tag']}: "
                f"vector {load_data['load_vector']}"
            )

        point_results = run_point_load_analysis(
            model_info=model_info,
            beam_info=beam_info,
            boundary_info=boundary_info,
            number_of_steps=NUMBER_OF_POINT_LOAD_STEPS,
            tolerance=1.0e-5,
            maximum_iterations=100,
        )
        final_point_step = point_results["history"][-1]

        print("\nPoint-load analysis results")
        print("---------------------------")
        print(f"Converged: {point_results['converged']}")
        print(
            "Final point-load factor: "
            f"{point_results['final_point_load_factor']:.6g}"
        )
        print(
            "Total horizontal support reaction: "
            f"{point_results['horizontal_support_reaction']:.6g} N"
        )
        print(
            "Total vertical support reaction: "
            f"{point_results['vertical_support_reaction']:.6g} N"
        )
        print(
            "Beam base moment: "
            f"{point_results['beam_base_moment_reaction']:.6g} N.mm"
        )
        print(
            "Maximum soil displacement: "
            f"{final_point_step['maximum_soil_displacement']:.6g} mm"
        )
        print(
            "Maximum-displacement node: "
            f"{final_point_step['maximum_displacement_node']}"
        )

        print("\nBeam local end forces")
        print("---------------------")
        print("Format: (Ni, Vi, Mi, Nj, Vj, Mj)")
        for element_tag, forces in point_results["beam_local_forces"].items():
            print(f"Beam {element_tag}: {forces}")
    else:
        print("\nNo point loads requested: POINT_LOADS is empty.")


if __name__ == "__main__":
    main()
