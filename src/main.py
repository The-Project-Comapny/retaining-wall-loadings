import gmsh
import pandas as pd
import ast
import numpy as np
import openseespy.opensees as ops
import math

##### User inputs #####
stp_location = r"C:\Users\MishalMohanlal\OneDrive - The Project Company\Documents\GitHub\retaining wall loadings\TIB BIN RAMP FEA 1.stp"
msh_location = r"C:\Users\MishalMohanlal\OneDrive - The Project Company\Documents\GitHub\retaining wall loadings\TIB BIN RAMP FEA 1.msh"


##### End user inputs #####

def create_mesh(path):

    gmsh.initialize()


    gmsh.option.setNumber("General.Terminal", 1)

    gmsh.model.add("model")

    # Import STEP geometry into OpenCASCADE
    gmsh.model.occ.importShapes(path)

    # Synchronize imported CAD geometry with Gmsh
    gmsh.model.occ.synchronize()

    # Mesh size limits
    gmsh.option.setNumber("Mesh.MeshSizeMin", 100)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 1000)

    # Generate 2D surface mesh
    gmsh.model.mesh.generate(2)

    # Recombine triangles into quadrilateral elements where possible
    gmsh.model.mesh.recombine()

    # Subdivision algorithm
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)

    # Refine mesh
    gmsh.model.mesh.refine()

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(msh_location)

    # Display model in Gmsh
    gmsh.fltk.run()


    gmsh.finalize()


def read_msh_to_dataframe(msh_path):
    """
    Read a Gmsh .msh file into a pandas DataFrame.

    Rows with record_type == "node" contain coordinates.
    Rows with record_type == "element" contain element type and connectivity.
    """
    gmsh.initialize()

    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.open(msh_path)

        records = []

        # Read nodes
        node_tags, coordinates, parametric_coordinates = (
            gmsh.model.mesh.getNodes()
        )

        for index, node_tag in enumerate(node_tags):
            records.append({
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
            })

        # Read elements grouped by their geometric entity
        for dimension, entity_tag in gmsh.model.getEntities():
            element_types, element_tags, node_tags_by_type = (
                gmsh.model.mesh.getElements(dimension, entity_tag)
            )

            for element_type, tags, connectivity in zip(
                element_types,
                element_tags,
                node_tags_by_type,
            ):
                (
                    element_name,
                    element_dimension,
                    element_order,
                    nodes_per_element,
                    local_coordinates,
                    primary_nodes,
                ) = gmsh.model.mesh.getElementProperties(element_type)

                for index, element_tag in enumerate(tags):
                    start = index * nodes_per_element
                    end = start + nodes_per_element

                    records.append({
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
                            int(node_tag)
                            for node_tag in connectivity[start:end]
                        ],
                    })

        return pd.DataFrame(records)

    finally:
        gmsh.finalize()

def create_opensees_sspquad_model(
    mesh_df,
    youngs_modulus,
    poissons_ratio,
    density,
    friction_angle,
    cohesion=0.0,
    dilation_angle=0.0,
    thickness=1.0,
    material_tag=1,
    coordinate_columns=("x", "z"),
    body_force=(0.0, -9.81),
    atmospheric_pressure=101.0,
):
    """
    Create a nonlinear plane-strain OpenSees soil model using:

        Element:  SSPquad
        Material: DruckerPrager

    Parameters
    ----------
    mesh_df : pandas.DataFrame
        DataFrame returned by read_msh_to_dataframe().

    youngs_modulus : float
        Initial Young's modulus of the soil.

    poissons_ratio : float
        Initial Poisson's ratio of the soil.

    density : float
        Soil mass density.

    friction_angle : float
        Mohr-Coulomb internal friction angle in degrees.

    cohesion : float, default=0.0
        Mohr-Coulomb cohesive intercept, in the same stress units
        as youngs_modulus.

    dilation_angle : float, default=0.0
        Plastic dilation angle in degrees.

        A value of zero gives non-associated, non-dilative plastic
        flow. The dilation angle should normally not exceed the
        friction angle.

    thickness : float, default=1.0
        Out-of-plane thickness.

    material_tag : int, default=1
        OpenSees material tag.

    coordinate_columns : tuple[str, str]
        Gmsh coordinate columns mapped to the OpenSees model plane.

        Examples:
            ("x", "z") if Z is vertical
            ("x", "y") if Y is vertical

    body_force : tuple[float, float]
        Body-force components in the two model directions.

    atmospheric_pressure : float, default=101.0
        Atmospheric pressure in the same stress units as the elastic
        modulus and cohesion.

    Returns
    -------
    dict
        Summary of the created OpenSees model.
    """

    # ----------------------------------------------------------
    # Input validation
    # ----------------------------------------------------------

    required_columns = {
        "record_type",
        "tag",
        "x",
        "y",
        "z",
        "connectivity",
    }

    missing_columns = required_columns.difference(mesh_df.columns)

    if missing_columns:
        raise ValueError(
            "mesh_df is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if youngs_modulus <= 0.0:
        raise ValueError("youngs_modulus must be greater than zero.")

    if not (-1.0 < poissons_ratio < 0.5):
        raise ValueError(
            "poissons_ratio must satisfy -1 < nu < 0.5."
        )

    if density < 0.0:
        raise ValueError("density cannot be negative.")

    if thickness <= 0.0:
        raise ValueError("thickness must be greater than zero.")

    if cohesion < 0.0:
        raise ValueError("cohesion cannot be negative.")

    if not (0.0 <= friction_angle < 90.0):
        raise ValueError(
            "friction_angle must satisfy 0 <= phi < 90 degrees."
        )

    if not (0.0 <= dilation_angle < 90.0):
        raise ValueError(
            "dilation_angle must satisfy 0 <= psi < 90 degrees."
        )

    if dilation_angle > friction_angle:
        raise ValueError(
            "dilation_angle should not exceed friction_angle."
        )

    if len(coordinate_columns) != 2:
        raise ValueError(
            "coordinate_columns must contain exactly two axes."
        )

    axis_1, axis_2 = coordinate_columns
    valid_axes = {"x", "y", "z"}

    if axis_1 not in valid_axes or axis_2 not in valid_axes:
        raise ValueError(
            "coordinate_columns must use 'x', 'y', or 'z'."
        )

    if axis_1 == axis_2:
        raise ValueError(
            "coordinate_columns must contain different axes."
        )

    if len(body_force) != 2:
        raise ValueError(
            "body_force must contain two components."
        )

    # ----------------------------------------------------------
    # Separate node and element data
    # ----------------------------------------------------------

    node_df = mesh_df.loc[
        mesh_df["record_type"].eq("node")
    ].copy()

    element_df = mesh_df.loc[
        mesh_df["record_type"].eq("element")
    ].copy()

    if node_df.empty:
        raise ValueError("No nodes were found in mesh_df.")

    if element_df.empty:
        raise ValueError("No elements were found in mesh_df.")

    node_df["tag"] = node_df["tag"].astype(int)
    element_df["tag"] = element_df["tag"].astype(int)

    if node_df["tag"].duplicated().any():
        raise ValueError("Duplicate node tags were found.")

    if element_df["tag"].duplicated().any():
        raise ValueError("Duplicate element tags were found.")

    # ----------------------------------------------------------
    # Extract model-plane coordinates
    # ----------------------------------------------------------

    node_coordinates = {}

    for _, row in node_df.iterrows():
        node_tag = int(row["tag"])

        coordinate_1 = float(row[axis_1])
        coordinate_2 = float(row[axis_2])

        if not np.isfinite(coordinate_1):
            raise ValueError(
                f"Node {node_tag} has an invalid {axis_1} coordinate."
            )

        if not np.isfinite(coordinate_2):
            raise ValueError(
                f"Node {node_tag} has an invalid {axis_2} coordinate."
            )

        node_coordinates[node_tag] = (
            coordinate_1,
            coordinate_2,
        )

    # ----------------------------------------------------------
    # Connectivity helpers
    # ----------------------------------------------------------

    def parse_connectivity(value):
        if isinstance(value, str):
            value = ast.literal_eval(value)

        if isinstance(value, np.ndarray):
            value = value.tolist()

        if not isinstance(value, (list, tuple)):
            raise TypeError(
                f"Invalid connectivity value: {value!r}"
            )

        return [int(node_tag) for node_tag in value]

    def signed_area(connectivity):
        coordinates = [
            node_coordinates[node_tag]
            for node_tag in connectivity
        ]

        twice_area = 0.0

        for index in range(4):
            x1, y1 = coordinates[index]
            x2, y2 = coordinates[(index + 1) % 4]

            twice_area += x1 * y2 - x2 * y1

        return 0.5 * twice_area

    # ----------------------------------------------------------
    # Extract four-node quadrilateral elements
    # ----------------------------------------------------------

    quad_elements = []
    ignored_element_tags = []

    for _, row in element_df.iterrows():
        element_tag = int(row["tag"])
        connectivity = parse_connectivity(row["connectivity"])

        # SSPquad accepts only four-node elements.
        if len(connectivity) != 4:
            ignored_element_tags.append(element_tag)
            continue

        missing_nodes = [
            node_tag
            for node_tag in connectivity
            if node_tag not in node_coordinates
        ]

        if missing_nodes:
            raise ValueError(
                f"Element {element_tag} references missing nodes: "
                f"{missing_nodes}"
            )

        if len(set(connectivity)) != 4:
            raise ValueError(
                f"Element {element_tag} contains repeated nodes."
            )

        area = signed_area(connectivity)

        if np.isclose(area, 0.0):
            raise ValueError(
                f"Element {element_tag} has zero projected area in "
                f"the {coordinate_columns} plane."
            )

        # SSPquad requires counter-clockwise nodes.
        if area < 0.0:
            connectivity = [
                connectivity[0],
                connectivity[3],
                connectivity[2],
                connectivity[1],
            ]

        quad_elements.append(
            {
                "element_tag": element_tag,
                "connectivity": connectivity,
            }
        )

    if not quad_elements:
        raise ValueError(
            "No valid four-node quadrilateral elements were found."
        )

    used_node_tags = sorted({
        node_tag
        for element in quad_elements
        for node_tag in element["connectivity"]
    })

    # ----------------------------------------------------------
    # Calculate Drucker-Prager parameters
    # ----------------------------------------------------------

    bulk_modulus = (
        youngs_modulus
        / (3.0 * (1.0 - 2.0 * poissons_ratio))
    )

    shear_modulus = (
        youngs_modulus
        / (2.0 * (1.0 + poissons_ratio))
    )

    phi = math.radians(friction_angle)
    psi = math.radians(dilation_angle)

    # Match the Drucker-Prager cone to Mohr-Coulomb under
    # triaxial compression.
    friction_parameter = (
        2.0 * math.sqrt(2.0) * math.sin(phi)
        / (
            math.sqrt(3.0)
            * (3.0 - math.sin(phi))
        )
    )

    yield_stress = (
        6.0 * cohesion * math.cos(phi)
        / (
            math.sqrt(2.0)
            * (3.0 - math.sin(phi))
        )
    )

    # Plastic volumetric response derived from dilation angle.
    dilation_parameter = (
        2.0 * math.sqrt(2.0) * math.sin(psi)
        / (
            math.sqrt(3.0)
            * (3.0 - math.sin(psi))
        )
    )

    # Perfectly plastic response: no hardening or softening.
    K_infinity = 0.0
    K_initial = 0.0
    delta_1 = 0.0
    delta_2 = 0.0
    hardening_modulus = 0.0
    hardening_type = 1.0

    # ----------------------------------------------------------
    # Create the OpenSees model
    # ----------------------------------------------------------

    ops.wipe()

    ops.model(
        "basic",
        "-ndm", 2,
        "-ndf", 2,
    )

    for node_tag in used_node_tags:
        coordinate_1, coordinate_2 = node_coordinates[node_tag]

        ops.node(
            node_tag,
            coordinate_1,
            coordinate_2,
        )

    # Nonlinear Drucker-Prager soil material.
    ops.nDMaterial(
        "DruckerPrager",
        int(material_tag),
        float(bulk_modulus),
        float(shear_modulus),
        float(yield_stress),
        float(friction_parameter),
        float(dilation_parameter),
        float(K_infinity),
        float(K_initial),
        float(delta_1),
        float(delta_2),
        float(hardening_modulus),
        float(hardening_type),
        float(density),
        float(atmospheric_pressure),
    )

    body_force_1 = float(body_force[0])
    body_force_2 = float(body_force[1])

    for element in quad_elements:
        ops.element(
            "SSPquad",
            int(element["element_tag"]),
            *element["connectivity"],
            int(material_tag),
            "PlaneStrain",
            float(thickness),
            body_force_1,
            body_force_2,
        )

    return {
        "number_of_nodes": len(used_node_tags),
        "number_of_elements": len(quad_elements),
        "node_tags": used_node_tags,
        "element_tags": [
            element["element_tag"]
            for element in quad_elements
        ],
        "ignored_element_tags": ignored_element_tags,
        "material_tag": int(material_tag),
        "material_type": "DruckerPrager",
        "element_type": "SSPquad",
        "formulation": "PlaneStrain",
        "bulk_modulus": bulk_modulus,
        "shear_modulus": shear_modulus,
        "yield_stress": yield_stress,
        "friction_parameter": friction_parameter,
        "dilation_parameter": dilation_parameter,
    }

def create_free_face_beam_elements(
    mesh_df,
    beam_youngs_modulus,
    beam_thickness,
    out_of_plane_width=1.0,
    transformation_tag=1,
    beam_node_start_tag=None,
    beam_element_start_tag=None,
    coordinate_tolerance=1.0e-6,
):
    """
    Create elastic beam elements along the vertical free face that:

        - starts at global coordinate (0, 0, 0);
        - follows the positive global Z-axis;
        - lies at global x = 0 and y = 0;
        - consists only of exterior SSPquad edges.

    The soil model must already have been created before calling this
    function.

    Separate three-DOF beam nodes are created at the same coordinates
    as the two-DOF soil nodes. The translations are connected using
    equalDOF, while beam rotation remains independent.

    Parameters
    ----------
    mesh_df : pandas.DataFrame
        DataFrame returned by read_msh_to_dataframe().

    beam_youngs_modulus : float
        Young's modulus of the beam or facing.

    beam_thickness : float
        Beam/facing thickness measured perpendicular to the free face
        in the model plane.

    out_of_plane_width : float, default=1.0
        Width perpendicular to the two-dimensional model plane.
        Use 1.0 for a unit-width retaining-wall model.

    transformation_tag : int, default=1
        Unique OpenSees geometric-transformation tag.

    beam_node_start_tag : int or None
        First tag used for the new beam nodes. If None, the function
        selects a tag above the largest existing OpenSees node tag.

    beam_element_start_tag : int or None
        First tag used for the new beam elements. If None, the function
        selects a tag above the largest existing OpenSees element tag.

    coordinate_tolerance : float
        Tolerance used to identify nodes lying on the global Z-axis.

    Returns
    -------
    dict
        Information about the created beam nodes and elements.
    """

    # ----------------------------------------------------------
    # Validate inputs
    # ----------------------------------------------------------

    required_columns = {
        "record_type",
        "tag",
        "x",
        "y",
        "z",
        "connectivity",
    }

    missing_columns = required_columns.difference(mesh_df.columns)

    if missing_columns:
        raise ValueError(
            "mesh_df is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if beam_youngs_modulus <= 0.0:
        raise ValueError(
            "beam_youngs_modulus must be greater than zero."
        )

    if beam_thickness <= 0.0:
        raise ValueError(
            "beam_thickness must be greater than zero."
        )

    if out_of_plane_width <= 0.0:
        raise ValueError(
            "out_of_plane_width must be greater than zero."
        )

    if coordinate_tolerance <= 0.0:
        raise ValueError(
            "coordinate_tolerance must be greater than zero."
        )

    # ----------------------------------------------------------
    # Separate node and element data
    # ----------------------------------------------------------

    node_df = mesh_df.loc[
        mesh_df["record_type"].eq("node")
    ].copy()

    element_df = mesh_df.loc[
        mesh_df["record_type"].eq("element")
    ].copy()

    if node_df.empty:
        raise ValueError("No node records were found.")

    if element_df.empty:
        raise ValueError("No element records were found.")

    node_df["tag"] = node_df["tag"].astype(int)
    element_df["tag"] = element_df["tag"].astype(int)

    node_coordinates_3d = {
        int(row["tag"]): (
            float(row["x"]),
            float(row["y"]),
            float(row["z"]),
        )
        for _, row in node_df.iterrows()
    }

    # ----------------------------------------------------------
    # Connectivity helper
    # ----------------------------------------------------------

    def parse_connectivity(value):
        if isinstance(value, str):
            value = ast.literal_eval(value)

        if isinstance(value, np.ndarray):
            value = value.tolist()

        if not isinstance(value, (list, tuple)):
            raise TypeError(
                f"Invalid connectivity value: {value!r}"
            )

        return [int(node_tag) for node_tag in value]

    # ----------------------------------------------------------
    # Find exterior edges
    # ----------------------------------------------------------

    edge_counts = {}

    for _, row in element_df.iterrows():
        connectivity = parse_connectivity(row["connectivity"])

        # Only four-node SSPquad-compatible elements.
        if len(connectivity) != 4:
            continue

        element_edges = [
            (connectivity[0], connectivity[1]),
            (connectivity[1], connectivity[2]),
            (connectivity[2], connectivity[3]),
            (connectivity[3], connectivity[0]),
        ]

        for node_i, node_j in element_edges:
            canonical_edge = tuple(sorted((node_i, node_j)))

            edge_counts[canonical_edge] = (
                edge_counts.get(canonical_edge, 0) + 1
            )

    # An edge belonging to only one quadrilateral is an exterior edge.
    exterior_edges = [
        edge
        for edge, count in edge_counts.items()
        if count == 1
    ]

    if not exterior_edges:
        raise ValueError(
            "No exterior quadrilateral edges were found."
        )

    # ----------------------------------------------------------
    # Select exterior edges on the positive global Z-axis
    # ----------------------------------------------------------

    def lies_on_free_face(node_tag):
        x, y, z = node_coordinates_3d[node_tag]

        return (
            abs(x) <= coordinate_tolerance
            and abs(y) <= coordinate_tolerance
            and z >= -coordinate_tolerance
        )

    free_face_edges = []

    for node_i, node_j in exterior_edges:
        if (
            lies_on_free_face(node_i)
            and lies_on_free_face(node_j)
        ):
            z_i = node_coordinates_3d[node_i][2]
            z_j = node_coordinates_3d[node_j][2]

            if np.isclose(
                z_i,
                z_j,
                atol=coordinate_tolerance,
                rtol=0.0,
            ):
                continue

            # Orient each beam segment from lower Z to higher Z.
            if z_i < z_j:
                free_face_edges.append((node_i, node_j))
            else:
                free_face_edges.append((node_j, node_i))

    if not free_face_edges:
        raise ValueError(
            "No exterior edges were found on the vertical line "
            "x = 0, y = 0, z >= 0."
        )

    # Sort segments from bottom to top.
    free_face_edges.sort(
        key=lambda edge: node_coordinates_3d[edge[0]][2]
    )

    free_face_soil_nodes = sorted(
        {
            node_tag
            for edge in free_face_edges
            for node_tag in edge
        },
        key=lambda node_tag: node_coordinates_3d[node_tag][2],
    )

    # ----------------------------------------------------------
    # Check that the face begins at the origin
    # ----------------------------------------------------------

    bottom_soil_node = free_face_soil_nodes[0]
    bottom_x, bottom_y, bottom_z = (
        node_coordinates_3d[bottom_soil_node]
    )

    if not (
        abs(bottom_x) <= coordinate_tolerance
        and abs(bottom_y) <= coordinate_tolerance
        and abs(bottom_z) <= coordinate_tolerance
    ):
        raise ValueError(
            "The detected free face does not start at (0, 0, 0). "
            f"Its lowest node is at "
            f"({bottom_x}, {bottom_y}, {bottom_z})."
        )

    # ----------------------------------------------------------
    # Check that the free face is one continuous chain
    # ----------------------------------------------------------

    expected_edges = {
        tuple(sorted((
            free_face_soil_nodes[index],
            free_face_soil_nodes[index + 1],
        )))
        for index in range(len(free_face_soil_nodes) - 1)
    }

    detected_edges = {
        tuple(sorted(edge))
        for edge in free_face_edges
    }

    if detected_edges != expected_edges:
        raise ValueError(
            "The detected free-face edges do not form one continuous "
            "vertical chain."
        )

    # ----------------------------------------------------------
    # Verify that the soil nodes exist in the OpenSees domain
    # ----------------------------------------------------------

    existing_node_tags = {
        int(tag)
        for tag in ops.getNodeTags()
    }

    missing_opensees_nodes = [
        node_tag
        for node_tag in free_face_soil_nodes
        if node_tag not in existing_node_tags
    ]

    if missing_opensees_nodes:
        raise ValueError(
            "The following free-face soil nodes do not exist in the "
            f"current OpenSees model: {missing_opensees_nodes}. "
            "Call create_opensees_sspquad_model() first."
        )

    # ----------------------------------------------------------
    # Generate unique node and element tags
    # ----------------------------------------------------------

    if beam_node_start_tag is None:
        beam_node_start_tag = max(existing_node_tags) + 1

    existing_element_tags = {
        int(tag)
        for tag in ops.getEleTags()
    }

    if beam_element_start_tag is None:
        if existing_element_tags:
            beam_element_start_tag = max(existing_element_tags) + 1
        else:
            beam_element_start_tag = 1

    proposed_beam_node_tags = set(
        range(
            beam_node_start_tag,
            beam_node_start_tag + len(free_face_soil_nodes),
        )
    )

    node_tag_conflicts = (
        proposed_beam_node_tags.intersection(existing_node_tags)
    )

    if node_tag_conflicts:
        raise ValueError(
            "The proposed beam node tags already exist: "
            f"{sorted(node_tag_conflicts)}"
        )

    number_of_beam_elements = len(free_face_edges)

    proposed_beam_element_tags = set(
        range(
            beam_element_start_tag,
            beam_element_start_tag
            + number_of_beam_elements,
        )
    )

    element_tag_conflicts = (
        proposed_beam_element_tags.intersection(existing_element_tags)
    )

    if element_tag_conflicts:
        raise ValueError(
            "The proposed beam element tags already exist: "
            f"{sorted(element_tag_conflicts)}"
        )

    # ----------------------------------------------------------
    # Calculate beam section properties
    # ----------------------------------------------------------

    cross_section_area = (
        out_of_plane_width * beam_thickness
    )

    second_moment_of_area = (
        out_of_plane_width
        * beam_thickness**3
        / 12.0
    )

    # ----------------------------------------------------------
    # Create three-DOF beam nodes
    # ----------------------------------------------------------

    # Existing soil nodes remain two-DOF nodes. Nodes created after
    # this command have UX, UZ and rotation.
    ops.model(
        "basic",
        "-ndm", 2,
        "-ndf", 3,
    )

    soil_to_beam_node = {}

    for index, soil_node_tag in enumerate(
        free_face_soil_nodes
    ):
        beam_node_tag = beam_node_start_tag + index

        x, y, z = node_coordinates_3d[soil_node_tag]

        # The global X-Z plane is mapped to OpenSees X-Y.
        ops.node(
            beam_node_tag,
            x,
            z,
        )

        # Couple horizontal and vertical translations.
        # Beam rotation remains independent.
        ops.equalDOF(
            soil_node_tag,
            beam_node_tag,
            1, 2,
        )

        soil_to_beam_node[soil_node_tag] = beam_node_tag

    # ----------------------------------------------------------
    # Create beam elements
    # ----------------------------------------------------------

    ops.geomTransf(
        "Linear",
        int(transformation_tag),
    )

    beam_element_tags = []
    beam_connectivity = {}

    for index, (soil_node_i, soil_node_j) in enumerate(
        free_face_edges
    ):
        beam_element_tag = beam_element_start_tag + index

        beam_node_i = soil_to_beam_node[soil_node_i]
        beam_node_j = soil_to_beam_node[soil_node_j]

        ops.element(
            "elasticBeamColumn",
            beam_element_tag,
            beam_node_i,
            beam_node_j,
            float(cross_section_area),
            float(beam_youngs_modulus),
            float(second_moment_of_area),
            int(transformation_tag),
        )

        beam_element_tags.append(beam_element_tag)

        beam_connectivity[beam_element_tag] = (
            beam_node_i,
            beam_node_j,
        )

    return {
        "number_of_beam_nodes": len(soil_to_beam_node),
        "number_of_beam_elements": len(beam_element_tags),
        "soil_node_tags": free_face_soil_nodes,
        "beam_node_tags": list(soil_to_beam_node.values()),
        "beam_element_tags": beam_element_tags,
        "soil_to_beam_node": soil_to_beam_node,
        "beam_connectivity": beam_connectivity,
        "cross_section_area": cross_section_area,
        "second_moment_of_area": second_moment_of_area,
        "transformation_tag": transformation_tag,
    }

def apply_model_boundary_conditions(
    mesh_df,
    model_info,
    beam_info,
    coordinate_tolerance=1.0e-6,
):
    """
    Apply boundary conditions to the soil and free-face beam.

    Soil boundary
    -------------
    All soil nodes on the positive global X-axis are fixed in both
    translations:

        y = 0
        z = 0
        x >= 0

    Beam boundary
    -------------
    The beam node at global coordinate (0, 0, 0) is effectively fixed
    in:

        horizontal translation
        vertical translation
        rotation

    Its translations are restrained through equalDOF because the
    coincident soil node is fixed. Therefore, only its rotational
    DOF is explicitly restrained here.

    Parameters
    ----------
    mesh_df : pandas.DataFrame
        DataFrame returned by read_msh_to_dataframe().

    model_info : dict
        Dictionary returned by create_opensees_sspquad_model().

    beam_info : dict
        Dictionary returned by create_free_face_beam_elements().

    coordinate_tolerance : float
        Coordinate tolerance for detecting nodes on the support line.

    Returns
    -------
    dict
        Information about the restrained soil and beam nodes.
    """

    # ----------------------------------------------------------
    # Validate inputs
    # ----------------------------------------------------------

    required_columns = {
        "record_type",
        "tag",
        "x",
        "y",
        "z",
    }

    missing_columns = required_columns.difference(mesh_df.columns)

    if missing_columns:
        raise ValueError(
            "mesh_df is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if coordinate_tolerance <= 0.0:
        raise ValueError(
            "coordinate_tolerance must be greater than zero."
        )

    if "node_tags" not in model_info:
        raise ValueError(
            "model_info does not contain 'node_tags'."
        )

    if "soil_to_beam_node" not in beam_info:
        raise ValueError(
            "beam_info does not contain 'soil_to_beam_node'."
        )

    soil_model_node_tags = {
        int(node_tag)
        for node_tag in model_info["node_tags"]
    }

    soil_to_beam_node = {
        int(soil_tag): int(beam_tag)
        for soil_tag, beam_tag
        in beam_info["soil_to_beam_node"].items()
    }

    existing_opensees_nodes = {
        int(node_tag)
        for node_tag in ops.getNodeTags()
    }

    # ----------------------------------------------------------
    # Extract soil-node coordinates
    # ----------------------------------------------------------

    node_df = mesh_df.loc[
        mesh_df["record_type"].eq("node")
    ].copy()

    node_df["tag"] = node_df["tag"].astype(int)

    node_coordinates = {
        int(row["tag"]): (
            float(row["x"]),
            float(row["y"]),
            float(row["z"]),
        )
        for _, row in node_df.iterrows()
        if int(row["tag"]) in soil_model_node_tags
    }

    missing_coordinates = (
        soil_model_node_tags.difference(node_coordinates)
    )

    if missing_coordinates:
        raise ValueError(
            "Coordinates were not found for the following soil nodes: "
            f"{sorted(missing_coordinates)}"
        )

    # ----------------------------------------------------------
    # Find soil support nodes along the positive global X-axis
    # ----------------------------------------------------------

    support_soil_nodes = []

    for node_tag, (x, y, z) in node_coordinates.items():
        is_on_support_line = (
            abs(y) <= coordinate_tolerance
            and abs(z) <= coordinate_tolerance
            and x >= -coordinate_tolerance
        )

        if is_on_support_line:
            support_soil_nodes.append(node_tag)

    if not support_soil_nodes:
        raise ValueError(
            "No soil support nodes were found along the line "
            "y = 0, z = 0, x >= 0."
        )

    # Sort support nodes from the origin along positive global X.
    support_soil_nodes.sort(
        key=lambda node_tag: node_coordinates[node_tag][0]
    )

    # ----------------------------------------------------------
    # Confirm that the support line starts at the origin
    # ----------------------------------------------------------

    origin_soil_nodes = []

    for node_tag in support_soil_nodes:
        x, y, z = node_coordinates[node_tag]

        if (
            abs(x) <= coordinate_tolerance
            and abs(y) <= coordinate_tolerance
            and abs(z) <= coordinate_tolerance
        ):
            origin_soil_nodes.append(node_tag)

    if not origin_soil_nodes:
        raise ValueError(
            "No soil node was found at the origin (0, 0, 0)."
        )

    # Find which origin soil node is connected to the free-face beam.
    beam_connected_origin_nodes = [
        node_tag
        for node_tag in origin_soil_nodes
        if node_tag in soil_to_beam_node
    ]

    if len(beam_connected_origin_nodes) == 0:
        raise ValueError(
            "The soil node at (0, 0, 0) is not connected to a "
            "free-face beam node."
        )

    if len(beam_connected_origin_nodes) > 1:
        raise ValueError(
            "More than one beam-connected soil node was found at "
            "(0, 0, 0)."
        )

    origin_soil_node = beam_connected_origin_nodes[0]
    origin_beam_node = soil_to_beam_node[origin_soil_node]

    # ----------------------------------------------------------
    # Verify OpenSees nodes exist
    # ----------------------------------------------------------

    missing_support_nodes = [
        node_tag
        for node_tag in support_soil_nodes
        if node_tag not in existing_opensees_nodes
    ]

    if missing_support_nodes:
        raise ValueError(
            "The following support nodes do not exist in the current "
            f"OpenSees domain: {missing_support_nodes}"
        )

    if origin_beam_node not in existing_opensees_nodes:
        raise ValueError(
            f"Beam node {origin_beam_node} does not exist in the "
            "current OpenSees domain."
        )

    # ----------------------------------------------------------
    # Apply soil restraints
    # ----------------------------------------------------------

    # Soil nodes have two DOFs:
    # DOF 1 = horizontal translation
    # DOF 2 = vertical translation
    for soil_node_tag in support_soil_nodes:
        ops.fix(
            soil_node_tag,
            1,  # Fix horizontal translation
            1,  # Fix vertical translation
        )

    # ----------------------------------------------------------
    # Apply beam-root restraint
    # ----------------------------------------------------------

    # The beam node has three DOFs:
    # DOF 1 = horizontal translation
    # DOF 2 = vertical translation
    # DOF 3 = rotation
    #
    # DOFs 1 and 2 are already restrained through:
    #
    # equalDOF(origin_soil_node, origin_beam_node, 1, 2)
    #
    # because the origin soil node was fixed above.
    # Therefore, explicitly restrain only beam rotation.
    ops.fix(
        origin_beam_node,
        0,  # Translation fixed through equalDOF
        0,  # Translation fixed through equalDOF
        1,  # Explicitly fix beam rotation
    )

    return {
        "support_soil_node_tags": support_soil_nodes,
        "number_of_support_soil_nodes": len(
            support_soil_nodes
        ),
        "origin_soil_node_tag": origin_soil_node,
        "origin_beam_node_tag": origin_beam_node,
        "soil_fixity": (1, 1),
        "beam_explicit_fixity": (0, 0, 1),
        "beam_effective_fixity": (1, 1, 1),
        "support_line": {
            "description": "Positive global X-axis",
            "start": (0.0, 0.0, 0.0),
            "conditions": "y = 0, z = 0, x >= 0",
        },
    }

def run_gravity_analysis(
    model_info,
    boundary_info=None,
    number_of_steps=20,
    tolerance=1.0e-8,
    maximum_iterations=50,
    time_series_tag=1,
    load_pattern_tag=1,
    hold_gravity_constant=True,
):
    """
    Run an incremental static gravity analysis.

    This function assumes that the SSPquad elements were created with
    their full body-force components, for example:

        body_force = (0.0, -density * 9.81)

    A SelfWeight element load is used so the stored SSPquad body force
    is increased incrementally from zero to its full value.

    Parameters
    ----------
    model_info : dict
        Dictionary returned by create_opensees_sspquad_model().

    boundary_info : dict or None
        Dictionary returned by apply_model_boundary_conditions().
        If supplied, support reactions are calculated after gravity.

    number_of_steps : int
        Number of gravity-load increments.

    tolerance : float
        Convergence tolerance for displacement increments.

    maximum_iterations : int
        Maximum equilibrium iterations in each load step.

    time_series_tag : int
        Unique OpenSees time-series tag.

    load_pattern_tag : int
        Unique OpenSees load-pattern tag.

    hold_gravity_constant : bool
        If True, gravity is held constant after convergence and the
        analysis time is reset to zero.

    Returns
    -------
    dict
        Gravity-analysis results and convergence information.
    """

    if "element_tags" not in model_info:
        raise ValueError(
            "model_info does not contain 'element_tags'."
        )

    if "node_tags" not in model_info:
        raise ValueError(
            "model_info does not contain 'node_tags'."
        )

    if number_of_steps <= 0:
        raise ValueError(
            "number_of_steps must be greater than zero."
        )

    if tolerance <= 0.0:
        raise ValueError(
            "tolerance must be greater than zero."
        )

    if maximum_iterations <= 0:
        raise ValueError(
            "maximum_iterations must be greater than zero."
        )

    soil_element_tags = [
        int(element_tag)
        for element_tag in model_info["element_tags"]
    ]

    soil_node_tags = [
        int(node_tag)
        for node_tag in model_info["node_tags"]
    ]

    existing_element_tags = {
        int(element_tag)
        for element_tag in ops.getEleTags()
    }

    missing_elements = [
        element_tag
        for element_tag in soil_element_tags
        if element_tag not in existing_element_tags
    ]

    if missing_elements:
        raise ValueError(
            "The following soil elements do not exist in the current "
            f"OpenSees model: {missing_elements}"
        )

    # Remove any previously constructed analysis objects.
    # This does not remove the model, loads, nodes or elements.
    ops.wipeAnalysis()

    # ----------------------------------------------------------
    # Create the incremental gravity load pattern
    # ----------------------------------------------------------

    ops.timeSeries(
        "Linear",
        int(time_series_tag),
    )

    ops.pattern(
        "Plain",
        int(load_pattern_tag),
        int(time_series_tag),
    )

    # SSPquad stores the body-force vector supplied during element
    # creation. SelfWeight factors of 1.0 apply the complete stored
    # X and vertical body-force components.
    #
    # Apply the element loads in groups to avoid an excessively long
    # OpenSees command for a large mesh.
    chunk_size = 500

    for start_index in range(
        0,
        len(soil_element_tags),
        chunk_size,
    ):
        element_chunk = soil_element_tags[
            start_index:start_index + chunk_size
        ]

        ops.eleLoad(
            "-ele",
            *element_chunk,
            "-type",
            "-SelfWeight",
            1.0,
            1.0,
        )

    # ----------------------------------------------------------
    # Configure the nonlinear static analysis
    # ----------------------------------------------------------

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

    gravity_increment = 1.0 / number_of_steps

    ops.integrator(
        "LoadControl",
        gravity_increment,
    )

    ops.analysis("Static")

    # ----------------------------------------------------------
    # Run gravity one increment at a time
    # ----------------------------------------------------------

    gravity_history = []

    for step in range(1, number_of_steps + 1):
        analysis_result = ops.analyze(1)

        if analysis_result != 0:
            current_load_factor = ops.getTime()

            raise RuntimeError(
                "Gravity analysis failed to converge.\n"
                f"Step: {step} of {number_of_steps}\n"
                f"Current load factor: {current_load_factor:.6g}\n"
                f"OpenSees return code: {analysis_result}\n"
                "Try increasing number_of_steps, increasing "
                "maximum_iterations, or reviewing the material and "
                "boundary conditions."
            )

        current_load_factor = float(ops.getTime())

        maximum_displacement = 0.0
        maximum_displacement_node = None

        for node_tag in soil_node_tags:
            displacement_x = float(
                ops.nodeDisp(node_tag, 1)
            )

            displacement_vertical = float(
                ops.nodeDisp(node_tag, 2)
            )

            displacement_magnitude = math.sqrt(
                displacement_x**2
                + displacement_vertical**2
            )

            if displacement_magnitude > maximum_displacement:
                maximum_displacement = displacement_magnitude
                maximum_displacement_node = node_tag

        gravity_history.append({
            "step": step,
            "load_factor": current_load_factor,
            "maximum_soil_displacement": maximum_displacement,
            "maximum_displacement_node": (
                maximum_displacement_node
            ),
        })

    # ----------------------------------------------------------
    # Calculate support reactions
    # ----------------------------------------------------------

    horizontal_reaction = None
    vertical_reaction = None
    beam_base_moment_reaction = None

    if boundary_info is not None:
        ops.reactions()

        support_soil_nodes = boundary_info[
            "support_soil_node_tags"
        ]

        horizontal_reaction = sum(
            float(ops.nodeReaction(node_tag, 1))
            for node_tag in support_soil_nodes
        )

        vertical_reaction = sum(
            float(ops.nodeReaction(node_tag, 2))
            for node_tag in support_soil_nodes
        )

        origin_beam_node = boundary_info[
            "origin_beam_node_tag"
        ]

        beam_base_moment_reaction = float(
            ops.nodeReaction(origin_beam_node, 3)
        )

    final_load_factor = float(ops.getTime())

    # ----------------------------------------------------------
    # Hold gravity for subsequent load stages
    # ----------------------------------------------------------

    if hold_gravity_constant:
        ops.loadConst(
            "-time",
            0.0,
        )

        # Remove only the analysis objects. The converged model state
        # and constant gravity load remain in the domain.
        ops.wipeAnalysis()

    return {
        "converged": True,
        "number_of_steps": number_of_steps,
        "final_gravity_load_factor": final_load_factor,
        "gravity_held_constant": hold_gravity_constant,
        "horizontal_support_reaction": horizontal_reaction,
        "vertical_support_reaction": vertical_reaction,
        "beam_base_moment_reaction": (
            beam_base_moment_reaction
        ),
        "history": gravity_history,
    }
create_mesh(stp_location)
mesh_df = read_msh_to_dataframe(msh_location)
print(mesh_df)

soil_properties = {
    "youngs_modulus": 30_000.0,  # kPa
    "poissons_ratio": 0.30,
    "density": 1.8,              # Mg/m³
    "friction_angle": 32.0,      # degrees
    "cohesion": 0.0,             # kPa
    "dilation_angle": 0.0,       # degrees
}


# ============================================================
# Consistent N-mm-s unit system
# ============================================================

# Soil input properties in commonly reported units
soil_E_kPa = 30_000.0
soil_density_Mg_m3 = 1.8
atmospheric_pressure_kPa = 101.0
cohesion_kPa = 1.0

# Conversion factors
kPa_to_N_mm2 = 0.001
Mg_m3_to_N_s2_mm4 = 1.0e-9
metres_to_mm = 1000.0

# Converted soil properties
soil_youngs_modulus = (
    soil_E_kPa * kPa_to_N_mm2
)

soil_density = (
    soil_density_Mg_m3
    * Mg_m3_to_N_s2_mm4
)

atmospheric_pressure = (
    atmospheric_pressure_kPa
    * kPa_to_N_mm2
)

soil_cohesion = (
    cohesion_kPa
    * kPa_to_N_mm2
)

# Gravity in mm/s²
gravity_acceleration = 9.81 * metres_to_mm

# SSPquad body force in N/mm³
soil_unit_weight = (
    soil_density * gravity_acceleration
)

print("Soil Young's modulus:", soil_youngs_modulus, "N/mm²")
print("Soil density:", soil_density, "N.s²/mm⁴")
print("Gravity:", gravity_acceleration, "mm/s²")
print("Soil unit weight:", soil_unit_weight, "N/mm³")

create_mesh(stp_location)

mesh_df = read_msh_to_dataframe(msh_location)

# Do not scale the geometry.
# Gmsh and OpenSees coordinates remain in millimetres.

model_info = create_opensees_sspquad_model(
    mesh_df=mesh_df,
    youngs_modulus=soil_youngs_modulus,  # 30 N/mm²
    poissons_ratio=0.30,
    density=soil_density,                # N.s²/mm⁴
    friction_angle=32.0,
    cohesion=soil_cohesion,              # N/mm²
    dilation_angle=0.0,
    thickness=1000.0,                    # 1 m model width
    material_tag=1,
    coordinate_columns=("x", "z"),
    body_force=(
        0.0,
        -soil_unit_weight,               # N/mm³
    ),
    atmospheric_pressure=(
        atmospheric_pressure             # 0.101 N/mm²
    ),
)

beam_info = create_free_face_beam_elements(
    mesh_df=mesh_df,
    beam_youngs_modulus=30_000.0,  # Concrete E, N/mm²
    beam_thickness=300.0,          # mm
    out_of_plane_width=1000.0,     # mm
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
    number_of_steps=100,
    tolerance=1.0e-7,
    maximum_iterations=100,
)