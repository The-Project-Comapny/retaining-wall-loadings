import json
import math
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QAction, QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


APP_NAME = "Retaining Wall Soil Analysis"
CONFIG_VERSION = 1
DEFAULT_PROJECT_DIR = Path(
    r"C:\Users\MishalMohanlal\OneDrive - The Project Company\Documents"
    r"\GitHub\retaining wall loadings"
)


class PathSelector(QWidget):
    path_changed = Signal(str)

    def __init__(self, caption, file_filter, directory=False, parent=None):
        super().__init__(parent)
        self.caption = caption
        self.file_filter = file_filter
        self.directory = directory

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.line_edit = QLineEdit()
        self.line_edit.setClearButtonEnabled(True)
        self.line_edit.textChanged.connect(self.path_changed)
        layout.addWidget(self.line_edit, 1)

        browse_button = QPushButton("Browse…")
        browse_button.clicked.connect(self.browse)
        layout.addWidget(browse_button)

    def text(self):
        return self.line_edit.text().strip()

    def setText(self, value):
        self.line_edit.setText(str(value or ""))

    @Slot()
    def browse(self):
        start = self.text() or str(DEFAULT_PROJECT_DIR)
        if self.directory:
            selected = QFileDialog.getExistingDirectory(
                self,
                self.caption,
                start,
            )
        else:
            selected, _ = QFileDialog.getOpenFileName(
                self,
                self.caption,
                start,
                self.file_filter,
            )
        if selected:
            self.setText(selected)


def make_double_spinbox(
    value,
    minimum=0.0,
    maximum=1.0e12,
    decimals=4,
    suffix="",
    step=None,
):
    widget = QDoubleSpinBox()
    widget.setDecimals(decimals)
    widget.setRange(minimum, maximum)
    widget.setValue(value)
    widget.setKeyboardTracking(True)
    if suffix:
        widget.setSuffix(f" {suffix}")
    if step is not None:
        widget.setSingleStep(step)
    widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return widget


def make_integer_spinbox(value, minimum=1, maximum=1_000_000, suffix=""):
    widget = QSpinBox()
    widget.setRange(minimum, maximum)
    widget.setValue(value)
    widget.setKeyboardTracking(True)
    if suffix:
        widget.setSuffix(f" {suffix}")
    widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return widget


class GeometryPreviewWorker(QObject):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        mesh_path,
        step_path=None,
        regenerate=False,
        mesh_size_min=10.0,
        mesh_size_max=100.0,
    ):
        super().__init__()
        self.mesh_path = str(mesh_path)
        self.step_path = str(step_path or "")
        self.regenerate = bool(regenerate)
        self.mesh_size_min = float(mesh_size_min)
        self.mesh_size_max = float(mesh_size_max)

    @staticmethod
    def _extract_current_model(gmsh_module):
        node_tags, coordinates, _ = gmsh_module.model.mesh.getNodes()
        node_tags = np.asarray(node_tags, dtype=np.int64)
        coordinates = np.asarray(coordinates, dtype=float).reshape((-1, 3))
        nodes = {
            int(tag): tuple(map(float, xyz))
            for tag, xyz in zip(node_tags, coordinates)
        }

        cells = []
        element_types, _, element_nodes = gmsh_module.model.mesh.getElements(2)
        for element_type, flattened_nodes in zip(
            element_types,
            element_nodes,
        ):
            properties = gmsh_module.model.mesh.getElementProperties(
                int(element_type)
            )
            number_of_nodes = int(properties[3])
            number_of_primary_nodes = int(properties[5])
            connectivity = np.asarray(
                flattened_nodes,
                dtype=np.int64,
            ).reshape((-1, number_of_nodes))

            if number_of_primary_nodes not in (3, 4):
                continue
            for row in connectivity:
                cells.append(
                    tuple(int(tag) for tag in row[:number_of_primary_nodes])
                )

        if not nodes:
            raise ValueError("The selected file contains no mesh nodes.")
        if not cells:
            raise ValueError(
                "No triangular or quadrilateral surface cells could be rendered."
            )
        return nodes, cells

    @staticmethod
    def _load_mesh_node_map(gmsh_module, mesh_path):
        if not mesh_path or not Path(mesh_path).is_file():
            return None
        gmsh_module.clear()
        gmsh_module.open(mesh_path)
        node_tags, coordinates, _ = gmsh_module.model.mesh.getNodes()
        coordinates = np.asarray(coordinates, dtype=float).reshape((-1, 3))
        return {
            int(tag): tuple(map(float, xyz))
            for tag, xyz in zip(node_tags, coordinates)
        }

    @Slot()
    def run(self):
        owns_gmsh = False
        try:
            try:
                import gmsh
            except ImportError as exc:
                raise RuntimeError(
                    "The Gmsh Python package is required to render MSH "
                    "files. Install the GUI requirements first."
                ) from exc

            if not gmsh.isInitialized():
                # The preview runs in a QThread. Avoid Gmsh's optional SIGINT
                # handler, which can only be installed by Python's main thread.
                gmsh.initialize(interruptible=False)
                owns_gmsh = True

            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.clear()

            if self.regenerate:
                if not Path(self.step_path).is_file():
                    raise FileNotFoundError(
                        f"STEP file not found: {self.step_path}"
                    )
                gmsh.model.add("retaining_wall_soil")
                imported = gmsh.model.occ.importShapes(self.step_path)
                if not imported:
                    raise ValueError("Gmsh did not import any STEP entities.")
                gmsh.model.occ.synchronize()
                gmsh.option.setNumber("Mesh.MeshSizeMin", self.mesh_size_min)
                gmsh.option.setNumber("Mesh.MeshSizeMax", self.mesh_size_max)
                gmsh.model.mesh.generate(2)
                gmsh.model.mesh.recombine()
                gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
                gmsh.model.mesh.refine()
                gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
                gmsh.write(self.mesh_path)
                gmsh.clear()

            if not Path(self.mesh_path).is_file():
                raise FileNotFoundError(
                    f"MSH file not found: {self.mesh_path}"
                )
            gmsh.open(self.mesh_path)
            nodes, cells = self._extract_current_model(gmsh)
            load_nodes = nodes
            node_source = "selected MSH file"

            result = {
                "source_kind": "MSH mesh",
                "nodes": nodes,
                "cells": cells,
                "load_nodes": load_nodes,
                "node_source": node_source,
                "regenerated": self.regenerate,
            }
            self.completed.emit(result)
        except Exception as exc:
            detail = "".join(
                traceback.format_exception_only(type(exc), exc)
            ).strip()
            self.failed.emit(detail)
        finally:
            try:
                if owns_gmsh:
                    gmsh.finalize()
            except Exception:
                pass


class AnalysisWorker(QObject):
    progress = Signal(int, str)
    diagnostic = Signal(object)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, configuration, solver_path):
        super().__init__()
        self.configuration = configuration
        self.solver_path = Path(solver_path)

    def _forward_convergence(self, diagnostic):
        data = dict(diagnostic)
        case_name = data.get("case_name", "")
        phase = data.get("phase", "")
        elastic_enabled = bool(
            self.configuration["soil"]["run_elastic_comparison"]
        )
        if elastic_enabled:
            ranges = {
                ("Elastic", "Gravity"): (20, 34),
                ("Elastic", "Applied loads"): (34, 55),
                ("Drucker-Prager", "Gravity"): (58, 72),
                ("Drucker-Prager", "Applied loads"): (72, 88),
            }
        else:
            ranges = {
                ("Drucker-Prager", "Gravity"): (20, 52),
                ("Drucker-Prager", "Applied loads"): (52, 88),
            }
        start, end = ranges.get((case_name, phase), (20, 88))
        total_steps = max(int(data.get("total_steps", 1)), 1)
        fraction = min(max(float(data.get("step", 0)) / total_steps, 0.0), 1.0)
        data["overall_percent"] = round(start + fraction * (end - start))
        self.diagnostic.emit(data)

    @staticmethod
    def _beam_profile(mesh_df, case):
        node_rows = mesh_df.loc[mesh_df["record_type"].eq("node")]
        coordinates = {
            int(row["tag"]): (float(row["x"]), float(row["z"]))
            for _, row in node_rows.iterrows()
        }
        beam_info = case["beam_info"]
        beam_to_soil = {
            int(beam_node): int(soil_node)
            for soil_node, beam_node
            in beam_info["soil_to_beam_node"].items()
        }
        segments = []
        for element_tag in beam_info["beam_element_tags"]:
            element_tag = int(element_tag)
            beam_i, beam_j = beam_info["beam_connectivity"][element_tag]
            z_i = coordinates[beam_to_soil[int(beam_i)]][1]
            z_j = coordinates[beam_to_soil[int(beam_j)]][1]
            forces = case["final_beam_local_forces"][element_tag]
            segments.append(
                {
                    "element_tag": element_tag,
                    "z_i": z_i,
                    "z_j": z_j,
                    "shear_i_kn": float(forces[1]) / 1000.0,
                    "shear_j_kn": -float(forces[4]) / 1000.0,
                    "moment_i_knm": float(forces[2]) / 1.0e6,
                    "moment_j_knm": -float(forces[5]) / 1.0e6,
                }
            )
        return sorted(segments, key=lambda segment: min(segment["z_i"], segment["z_j"]))

    @staticmethod
    def _case_summary(solver, case):
        final = case["final_results"]
        final_step = final["history"][-1]
        maximum_shear, maximum_moment = solver._maximum_beam_actions(
            case["final_beam_local_forces"]
        )
        gravity = case["gravity_results"]
        expected_weight = float(case["expected_soil_weight"])
        vertical_reaction = abs(float(gravity["vertical_support_reaction"]))
        equilibrium_error = vertical_reaction - expected_weight
        return {
            "case_name": case["case_name"],
            "converged": bool(final["converged"]),
            "maximum_soil_displacement_mm": float(
                final_step["maximum_soil_displacement"]
            ),
            "maximum_displacement_node": int(
                final_step["maximum_displacement_node"]
            ),
            "horizontal_reaction_n": float(final["horizontal_support_reaction"]),
            "vertical_reaction_n": float(final["vertical_support_reaction"]),
            "beam_base_moment_nmm": float(final["beam_base_moment_reaction"]),
            "maximum_beam_shear_n": float(maximum_shear),
            "maximum_beam_moment_nmm": float(maximum_moment),
            "expected_soil_weight_n": expected_weight,
            "gravity_equilibrium_error_n": equilibrium_error,
            "gravity_equilibrium_error_percent": (
                100.0 * equilibrium_error / expected_weight
                if expected_weight else math.nan
            ),
        }

    @Slot()
    def run(self):
        try:
            import importlib.util

            self.progress.emit(3, "Loading analysis engine")
            spec = importlib.util.spec_from_file_location(
                "retaining_wall_model_gui",
                self.solver_path,
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot load solver: {self.solver_path}")
            solver = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(solver)

            geometry = self.configuration["geometry"]
            soil = self.configuration["soil"]
            beam = self.configuration["beam"]
            analysis = self.configuration["analysis"]
            output = self.configuration["output"]
            mesh_path = Path(geometry["mesh_path"])

            if geometry["recreate_mesh"]:
                self.progress.emit(8, "Creating quadrilateral Gmsh mesh")
                solver.create_mesh(
                    geometry["step_path"],
                    str(mesh_path),
                    mesh_size_min=geometry["mesh_size_min_mm"],
                    mesh_size_max=geometry["mesh_size_max_mm"],
                    show_gui=False,
                )

            self.progress.emit(18, "Reading mesh and checking load nodes")
            mesh_df = solver.read_msh_to_dataframe(str(mesh_path))

            node_rows = mesh_df.loc[mesh_df["record_type"].eq("node")]
            soil_nodes = {
                int(row["tag"]): (
                    float(row["x"]),
                    float(row["y"]),
                    float(row["z"]),
                )
                for _, row in node_rows.iterrows()
            }
            tolerance = float(beam["coordinate_tolerance_mm"])
            face_soil_nodes = sorted(
                (
                    tag
                    for tag, (x, y, z) in soil_nodes.items()
                    if abs(x) <= tolerance
                    and abs(y) <= tolerance
                    and z >= -tolerance
                ),
                key=lambda tag: soil_nodes[tag][2],
            )
            beam_start = max(soil_nodes) + 1
            predicted_beam_nodes = {
                beam_start + index for index in range(len(face_soil_nodes))
            }
            point_loads = []
            for index, load in enumerate(self.configuration["point_loads"], start=1):
                tag = int(load["node_tag"])
                target = str(load["target"]).strip().lower()
                if target == "soil" and tag not in soil_nodes:
                    raise ValueError(f"Point load {index}: soil node {tag} is absent.")
                if target == "beam" and tag not in predicted_beam_nodes:
                    raise ValueError(f"Point load {index}: beam node {tag} is absent.")
                if target == "soil" and load["direction"] == "moment":
                    raise ValueError(
                        f"Point load {index}: a moment requires a beam node."
                    )
                point_loads.append(
                    (tag, float(load["magnitude"]), str(load["direction"]))
                )

            density = soil["soil_density_mg_m3"] * solver.MG_M3_TO_N_S2_MM4
            gravity_acceleration = analysis["gravity_m_s2"] * solver.M_TO_MM
            unit_weight = density * gravity_acceleration
            common = {
                "mesh_df": mesh_df,
                "youngs_modulus": soil["soil_e_kpa"] * solver.KPA_TO_N_MM2,
                "poissons_ratio": soil["soil_poissons_ratio"],
                "density": density,
                "friction_angle": soil["friction_angle_deg"],
                "cohesion": soil["cohesion_kpa"] * solver.KPA_TO_N_MM2,
                "dilation_angle": soil["dilation_angle_deg"],
                "thickness": geometry["model_width_mm"],
                "body_force": (0.0, -unit_weight),
                "atmospheric_pressure": (
                    soil["atmospheric_pressure_kpa"] * solver.KPA_TO_N_MM2
                ),
                "hardening_ratio": soil["hardening_ratio"],
                "beam_youngs_modulus": beam["beam_e_n_mm2"],
                "beam_thickness": beam["beam_thickness_mm"],
                "beam_width": beam["beam_width_mm"],
                "point_loads": point_loads,
                "line_loads": self.configuration["line_loads"],
                "number_of_gravity_steps": analysis["gravity_steps"],
                "number_of_point_load_steps": analysis["point_load_steps"],
                "coordinate_tolerance": tolerance,
                "tolerance": analysis["convergence_tolerance"],
                "maximum_iterations": analysis["maximum_iterations"],
                "progress_callback": self._forward_convergence,
            }

            elastic_case = None
            if soil["run_elastic_comparison"]:
                self.progress.emit(30, "Solving ElasticIsotropic comparison")
                elastic_case = solver.run_complete_analysis_case(
                    case_name="Elastic",
                    material_model="elastic",
                    **common,
                )
                self.progress.emit(58, "Elastic model converged")

            self.progress.emit(62, "Solving Drucker-Prager model")
            nonlinear_case = solver.run_complete_analysis_case(
                case_name="Drucker-Prager",
                material_model="drucker_prager",
                **common,
            )
            self.progress.emit(88, "Preparing comparison results")

            cases = [case for case in (elastic_case, nonlinear_case) if case]
            comparison_records = []
            comparison = None
            if elastic_case is not None:
                comparison = solver.create_elastic_nonlinear_comparison(
                    elastic_case,
                    nonlinear_case,
                )
                comparison_records = comparison.reset_index().to_dict("records")

            summaries = [self._case_summary(solver, case) for case in cases]
            profiles = {
                case["case_name"]: self._beam_profile(mesh_df, case)
                for case in cases
            }
            convergence_records = []
            for case in cases:
                for result_name in ("gravity_results", "point_results"):
                    phase_results = case.get(result_name)
                    if phase_results is None:
                        continue
                    for diagnostic in phase_results["history"]:
                        convergence_records.append(
                            {
                                "case": case["case_name"],
                                "phase": diagnostic["phase"],
                                "step": diagnostic["step"],
                                "total_steps": diagnostic["total_steps"],
                                "load_factor": diagnostic["load_factor"],
                                "iterations": diagnostic["iterations"],
                                "convergence_norm": diagnostic[
                                    "convergence_norm"
                                ],
                                "tolerance": diagnostic["tolerance"],
                                "maximum_soil_displacement_mm": diagnostic[
                                    "maximum_soil_displacement"
                                ],
                            }
                        )

            output_directory = Path(output["output_directory"])
            output_directory.mkdir(parents=True, exist_ok=True)
            (output_directory / "analysis_configuration.json").write_text(
                json.dumps(self.configuration, indent=2),
                encoding="utf-8",
            )
            if comparison is not None:
                comparison.to_csv(
                    output_directory / "elastic_vs_drucker_prager.csv"
                )
            summary_frame = solver.pd.DataFrame(summaries)
            summary_frame.to_csv(output_directory / "analysis_summary.csv", index=False)
            solver.pd.DataFrame(convergence_records).to_csv(
                output_directory / "convergence_history.csv",
                index=False,
            )

            self.progress.emit(100, "Analysis complete")
            self.completed.emit(
                {
                    "configuration": self.configuration,
                    "mesh_path": str(mesh_path),
                    "cases": cases,
                    "summaries": summaries,
                    "profiles": profiles,
                    "comparison_records": comparison_records,
                    "output_directory": str(output_directory),
                }
            )
        except Exception:
            self.failed.emit(traceback.format_exc())


class ModelPreviewCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(9, 7), constrained_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)
        self.geometry_data = None
        self.loads = []
        self.line_loads = []
        self.show_node_tags = True

    def set_geometry(self, geometry_data):
        self.geometry_data = geometry_data
        self.redraw()

    def set_loads(self, loads, line_loads=None):
        self.loads = list(loads)
        if line_loads is not None:
            self.line_loads = list(line_loads)
        self.redraw()

    def set_show_node_tags(self, enabled):
        self.show_node_tags = bool(enabled)
        self.redraw()

    @staticmethod
    def _boundary_node_tags(cells):
        edge_counts = Counter()
        for cell in cells:
            for index, node_i in enumerate(cell):
                node_j = cell[(index + 1) % len(cell)]
                edge_counts[tuple(sorted((node_i, node_j)))] += 1
        return {
            tag
            for edge, count in edge_counts.items()
            if count == 1
            for tag in edge
        }

    @staticmethod
    def _beam_node_map(load_nodes):
        if not load_nodes:
            return {}
        x_values = [xyz[0] for xyz in load_nodes.values()]
        z_values = [xyz[2] for xyz in load_nodes.values()]
        span = max(max(x_values) - min(x_values), max(z_values) - min(z_values))
        tolerance = max(1.0e-6, span * 1.0e-8)
        face_soil_nodes = sorted(
            (
                tag
                for tag, (x, y, z) in load_nodes.items()
                if abs(x) <= tolerance
                and abs(y) <= tolerance
                and z >= -tolerance
            ),
            key=lambda tag: load_nodes[tag][2],
        )
        if not face_soil_nodes:
            return {}
        beam_start = max(load_nodes) + 1
        return {
            beam_start + index: soil_tag
            for index, soil_tag in enumerate(face_soil_nodes)
        }

    def _load_coordinate(self, load, load_nodes, beam_map):
        node_tag = int(load["node_tag"])
        if load["target"] == "Beam":
            soil_tag = beam_map.get(node_tag)
            if soil_tag is None:
                return None
            return load_nodes.get(soil_tag)
        return load_nodes.get(node_tag)

    def redraw(self):
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.set_facecolor("#f7f9fc")

        if self.geometry_data is None:
            axis.text(
                0.5,
                0.54,
                "Select an MSH file and click Render MSH",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=13,
                color="#526070",
            )
            axis.text(
                0.5,
                0.46,
                "Point loads will appear here as they are entered",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="#7a8794",
            )
            axis.set_xticks([])
            axis.set_yticks([])
            self.draw_idle()
            return

        nodes = self.geometry_data["nodes"]
        cells = self.geometry_data["cells"]
        load_nodes = self.geometry_data["load_nodes"]
        polygons = [
            [(nodes[tag][0], nodes[tag][2]) for tag in cell]
            for cell in cells
        ]
        collection = PolyCollection(
            polygons,
            facecolors="#dce9df",
            edgecolors="#50766a",
            linewidths=0.45,
            alpha=0.82,
        )
        axis.add_collection(collection)

        display_x = [xyz[0] for xyz in nodes.values()]
        display_z = [xyz[2] for xyz in nodes.values()]
        axis.update_datalim(np.column_stack((display_x, display_z)))
        axis.autoscale_view()
        x_span = max(display_x) - min(display_x)
        z_span = max(display_z) - min(display_z)
        overall_span = max(x_span, z_span, 1.0)
        padding = 0.06 * overall_span
        axis.set_xlim(min(display_x) - padding, max(display_x) + padding)
        axis.set_ylim(min(display_z) - padding, max(display_z) + padding)

        boundary_tags = self._boundary_node_tags(cells)
        boundary_points = [
            nodes[tag] for tag in boundary_tags if tag in nodes
        ]
        if boundary_points:
            axis.scatter(
                [point[0] for point in boundary_points],
                [point[2] for point in boundary_points],
                s=10,
                color="#1d4d40",
                zorder=4,
            )

        if self.show_node_tags:
            tag_source = load_nodes
            if len(tag_source) > 300:
                # Large meshes stay legible by labelling only nodes on the
                # exterior boundaries, including the supported/free faces.
                min_x = min(xyz[0] for xyz in tag_source.values())
                min_z = min(xyz[2] for xyz in tag_source.values())
                tolerance = overall_span * 1.0e-7
                label_tags = [
                    tag
                    for tag, xyz in tag_source.items()
                    if abs(xyz[0] - min_x) <= tolerance
                    or abs(xyz[2] - min_z) <= tolerance
                ]
            else:
                label_tags = list(tag_source)

            for tag in label_tags:
                x, _, z = tag_source[tag]
                axis.annotate(
                    str(tag),
                    (x, z),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6,
                    color="#25364a",
                    zorder=7,
                )

        beam_map = self._beam_node_map(load_nodes)
        arrow_length = 0.10 * overall_span
        invalid_loads = []

        for index, load in enumerate(self.loads, start=1):
            coordinate = self._load_coordinate(load, load_nodes, beam_map)
            if coordinate is None:
                invalid_loads.append(index)
                continue

            x, _, z = coordinate
            magnitude = float(load["magnitude"])
            direction = load["direction"]
            if magnitude == 0.0:
                continue

            if direction in ("x", "z"):
                sign = 1.0 if magnitude > 0.0 else -1.0
                dx = sign * arrow_length if direction == "x" else 0.0
                dz = sign * arrow_length if direction == "z" else 0.0
                start = (x - dx, z - dz)
                axis.annotate(
                    "",
                    xy=(x, z),
                    xytext=start,
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": "#d84335",
                        "linewidth": 2.2,
                        "mutation_scale": 14,
                    },
                    zorder=10,
                )
                axis.annotate(
                    f"L{index}: {magnitude / 1000.0:.4g} kN",
                    start,
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                    color="#b3261e",
                    zorder=11,
                )
            elif direction == "moment":
                radius = 0.07 * overall_span
                if magnitude > 0.0:
                    start = (x + radius, z)
                    end = (x, z + radius)
                    connection = "arc3,rad=0.55"
                else:
                    start = (x, z + radius)
                    end = (x + radius, z)
                    connection = "arc3,rad=-0.55"
                axis.add_patch(
                    FancyArrowPatch(
                        start,
                        end,
                        connectionstyle=connection,
                        arrowstyle="-|>",
                        mutation_scale=14,
                        color="#8c2d7a",
                        linewidth=2.2,
                        zorder=10,
                    )
                )
                axis.annotate(
                    f"L{index}: {magnitude / 1.0e6:.4g} kN.m",
                    (x + radius, z + radius),
                    fontsize=8,
                    fontweight="bold",
                    color="#7a1f68",
                    zorder=11,
                )

        for index, line_load in enumerate(self.line_loads, start=1):
            start_load = {
                "target": line_load["target"],
                "node_tag": line_load["start_node"],
            }
            end_load = {
                "target": line_load["target"],
                "node_tag": line_load["end_node"],
            }
            start_coordinate = self._load_coordinate(
                start_load,
                load_nodes,
                beam_map,
            )
            end_coordinate = self._load_coordinate(
                end_load,
                load_nodes,
                beam_map,
            )
            if start_coordinate is None or end_coordinate is None:
                invalid_loads.append(f"DL{index}")
                continue

            x_start, _, z_start = start_coordinate
            x_end, _, z_end = end_coordinate
            q_start = float(line_load["start_magnitude"])
            q_end = float(line_load["end_magnitude"])
            direction = line_load["direction"]
            samples = 7

            axis.plot(
                (x_start, x_end),
                (z_start, z_end),
                color="#e38b00",
                linewidth=2.0,
                zorder=8,
            )
            maximum_q = max(abs(q_start), abs(q_end), 1.0)
            for sample in range(samples):
                ratio = sample / (samples - 1)
                x = x_start + ratio * (x_end - x_start)
                z = z_start + ratio * (z_end - z_start)
                q_value = q_start + ratio * (q_end - q_start)
                if q_value == 0.0:
                    continue
                scaled_length = arrow_length * (
                    0.35 + 0.65 * abs(q_value) / maximum_q
                )
                sign = 1.0 if q_value > 0.0 else -1.0
                dx = sign * scaled_length if direction == "x" else 0.0
                dz = sign * scaled_length if direction == "z" else 0.0
                axis.annotate(
                    "",
                    xy=(x, z),
                    xytext=(x - dx, z - dz),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": "#e38b00",
                        "linewidth": 1.7,
                        "mutation_scale": 11,
                    },
                    zorder=9,
                )
            axis.annotate(
                f"DL{index}: {q_start:.4g} to {q_end:.4g} N/mm",
                ((x_start + x_end) / 2.0, (z_start + z_end) / 2.0),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                color="#a45f00",
                zorder=11,
            )

        title = self.geometry_data["source_kind"]
        if invalid_loads:
            title += f" — load rows without matching nodes: {invalid_loads}"
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.set_xlabel("Global X (mm)")
        axis.set_ylabel("Global Z (mm)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, color="#aab7c4", alpha=0.22)
        self.draw_idle()


class AnalysisResultsCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(10, 7), constrained_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)
        self.show_message("Run an analysis to display beam-force diagrams")

    def show_message(self, message):
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.text(
            0.5,
            0.5,
            message,
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="#526070",
            fontsize=12,
        )
        axis.set_axis_off()
        self.draw_idle()

    def set_results(self, result):
        profiles = result["profiles"]
        case_names = list(profiles)
        self.figure.clear()
        axes = self.figure.subplots(
            2,
            len(case_names),
            squeeze=False,
            sharey=True,
        )
        all_shear = [0.0]
        all_moment = [0.0]

        for column, case_name in enumerate(case_names):
            shear_axis = axes[0, column]
            moment_axis = axes[1, column]
            for segment in profiles[case_name]:
                z_values = (segment["z_i"], segment["z_j"])
                shear_values = (
                    segment["shear_i_kn"],
                    segment["shear_j_kn"],
                )
                moment_values = (
                    segment["moment_i_knm"],
                    segment["moment_j_knm"],
                )
                all_shear.extend(shear_values)
                all_moment.extend(moment_values)
                shear_axis.plot(shear_values, z_values, color="#187a72", linewidth=2)
                shear_axis.fill_betweenx(
                    z_values,
                    (0.0, 0.0),
                    shear_values,
                    color="#4db6ac",
                    alpha=0.28,
                )
                moment_axis.plot(
                    moment_values,
                    z_values,
                    color="#7756a5",
                    linewidth=2,
                )
                moment_axis.fill_betweenx(
                    z_values,
                    (0.0, 0.0),
                    moment_values,
                    color="#9b7bc4",
                    alpha=0.28,
                )

            shear_axis.set_title(f"{case_name} — shear")
            moment_axis.set_title(f"{case_name} — moment")
            shear_axis.set_xlabel("V (kN)")
            moment_axis.set_xlabel("M (kN.m)")
            shear_axis.set_ylabel("Global Z (mm)")
            moment_axis.set_ylabel("Global Z (mm)")
            for axis in (shear_axis, moment_axis):
                axis.axvline(0.0, color="#2d2d2d", linewidth=0.9)
                axis.grid(True, alpha=0.25)

        shear_limit = max(abs(value) for value in all_shear)
        moment_limit = max(abs(value) for value in all_moment)
        for column in range(len(case_names)):
            if shear_limit > 0.0:
                axes[0, column].set_xlim(-1.15 * shear_limit, 1.15 * shear_limit)
            if moment_limit > 0.0:
                axes[1, column].set_xlim(-1.15 * moment_limit, 1.15 * moment_limit)

        self.figure.suptitle(
            "Free-face beam actions after gravity and applied loads",
            fontsize=14,
            fontweight="bold",
        )
        self.draw_idle()


class RetainingWallApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 900)
        self.setMinimumSize(1180, 720)

        self.fields = {}
        self.load_rows = []
        self.line_load_rows = []
        self.geometry_data = None
        self.preview_thread = None
        self.preview_worker = None
        self.analysis_thread = None
        self.analysis_worker = None
        self.last_analysis_result = None
        self.convergence_row_keys = {}
        self._last_convergence_log = {}
        self._refresh_mesh_after_analysis = False
        self.redraw_timer = QTimer(self)
        self.redraw_timer.setSingleShot(True)
        self.redraw_timer.setInterval(80)
        self.redraw_timer.timeout.connect(self.update_live_load_preview)

        self._build_actions()
        self._build_interface()
        self._apply_style()
        self._set_defaults()
        self.add_load_row(
            {
                "target": "Soil",
                "node_tag": 7,
                "magnitude": -100_000.0,
                "direction": "z",
            }
        )

        self.preview_canvas.redraw()
        self.log(
            "System ready. Render the MSH model, define the loads, then "
            "select Run Analysis."
        )

    def _build_actions(self):
        self.new_action = QAction("New", self)
        self.new_action.setShortcut("Ctrl+N")
        self.new_action.triggered.connect(self.reset_configuration)

        self.open_action = QAction("Open Configuration…", self)
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.triggered.connect(self.load_configuration)

        self.save_action = QAction("Save Configuration…", self)
        self.save_action.setShortcut("Ctrl+S")
        self.save_action.triggered.connect(self.save_configuration)

        toolbar = QToolBar("Project", self)
        toolbar.setMovable(False)
        toolbar.addAction(self.new_action)
        toolbar.addAction(self.open_action)
        toolbar.addAction(self.save_action)
        self.addToolBar(toolbar)

    def _build_interface(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        header = QFrame()
        header.setObjectName("HeaderFrame")
        header_layout = QHBoxLayout(header)
        title_layout = QVBoxLayout()
        title = QLabel(APP_NAME)
        title.setObjectName("AppTitle")
        subtitle = QLabel(
            "SSPquad soil and bonded free-face beam — N, mm and seconds"
        )
        subtitle.setObjectName("AppSubtitle")
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        header_layout.addLayout(title_layout)
        header_layout.addStretch(1)

        self.validate_button = QPushButton("Validate Inputs")
        self.validate_button.clicked.connect(self.validate_inputs)
        header_layout.addWidget(self.validate_button)

        self.analysis_progress = QProgressBar()
        self.analysis_progress.setRange(0, 100)
        self.analysis_progress.setFixedWidth(210)
        self.analysis_progress.setTextVisible(True)
        self.analysis_progress.setVisible(False)
        header_layout.addWidget(self.analysis_progress)

        self.run_button = QPushButton("Run Analysis")
        self.run_button.setToolTip(
            "Run gravity, all applied loads and the selected material cases."
        )
        self.run_button.clicked.connect(self.run_analysis)
        header_layout.addWidget(self.run_button)
        root_layout.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter, 1)

        self.tabs = QTabWidget()
        # Load tables contain numeric spin boxes with their own arrow controls.
        # Keep enough width for the entered values to remain readable.
        self.tabs.setMinimumWidth(680)
        self.tabs.addTab(self._build_geometry_tab(), "Geometry & Mesh")
        self.tabs.addTab(self._build_soil_tab(), "Soil")
        self.tabs.addTab(self._build_beam_tab(), "Beam")
        self.tabs.addTab(self._build_loads_tab(), "Point Loads")
        self.tabs.addTab(self._build_analysis_tab(), "Analysis & Output")
        splitter.addWidget(self.tabs)

        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        preview_toolbar = QHBoxLayout()
        preview_toolbar.addWidget(QLabel("Mesh preview:"))
        self.render_button = QPushButton("Render MSH")
        self.render_button.clicked.connect(self.render_preview)
        preview_toolbar.addWidget(self.render_button)
        self.show_nodes_checkbox = QCheckBox("Show node tags")
        self.show_nodes_checkbox.setChecked(True)
        self.show_nodes_checkbox.toggled.connect(
            self.preview_canvas_show_nodes
        )
        preview_toolbar.addWidget(self.show_nodes_checkbox)
        preview_toolbar.addStretch(1)
        preview_layout.addLayout(preview_toolbar)

        self.preview_canvas = ModelPreviewCanvas()
        preview_layout.addWidget(
            NavigationToolbar2QT(self.preview_canvas, self),
        )
        preview_layout.addWidget(self.preview_canvas, 1)

        self.preview_information = QLabel(
            "No geometry loaded. MSH node tags are used for exact load placement."
        )
        self.preview_information.setWordWrap(True)
        self.preview_information.setObjectName("InformationLabel")
        preview_layout.addWidget(self.preview_information)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(135)
        self.log_output.setPlaceholderText("Status and validation messages")
        preview_layout.addWidget(self.log_output)

        self.output_tabs = QTabWidget()
        self.output_tabs.addTab(preview_panel, "Model & Loads")
        self.output_tabs.addTab(self._build_results_panel(), "Results")
        splitter.addWidget(self.output_tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([700, 760])

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

    def _build_results_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)

        self.result_status = QLabel("No analysis results are available yet.")
        self.result_status.setWordWrap(True)
        self.result_status.setObjectName("InformationLabel")
        layout.addWidget(self.result_status)

        self.results_table = QTableWidget(0, 6)
        self.results_table.setHorizontalHeaderLabels(
            [
                "Result",
                "Unit",
                "Elastic",
                "Drucker-Prager",
                "Difference",
                "Difference (%)",
            ]
        )
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        results_header = self.results_table.horizontalHeader()
        results_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, 6):
            results_header.setSectionResizeMode(
                column,
                QHeaderView.ResizeMode.ResizeToContents,
            )
        self.results_table.setMaximumHeight(255)
        layout.addWidget(self.results_table)

        self.results_canvas = AnalysisResultsCanvas()
        layout.addWidget(NavigationToolbar2QT(self.results_canvas, self))
        layout.addWidget(self.results_canvas, 1)
        return panel

    def _scrollable_tab(self, content):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        return scroll

    def _build_geometry_tab(self):
        content = QWidget()
        layout = QVBoxLayout(content)

        files_group = QGroupBox("CAD and mesh files")
        files_form = QFormLayout(files_group)
        self.step_selector = PathSelector(
            "Select STEP geometry",
            "STEP files (*.step *.stp);;All files (*.*)",
        )
        self.mesh_selector = PathSelector(
            "Select Gmsh mesh",
            "Gmsh mesh files (*.msh);;All files (*.*)",
        )
        self.fields["step_path"] = self.step_selector
        self.fields["mesh_path"] = self.mesh_selector
        files_form.addRow("STEP file", self.step_selector)
        files_form.addRow("MSH file", self.mesh_selector)

        self.recreate_mesh = QCheckBox(
            "Regenerate MSH from STEP when rendering or running analysis"
        )
        self.fields["recreate_mesh"] = self.recreate_mesh
        files_form.addRow("", self.recreate_mesh)
        layout.addWidget(files_group)

        mesh_group = QGroupBox("Mesh and model dimensions")
        mesh_form = QFormLayout(mesh_group)
        self.fields["mesh_size_min_mm"] = make_double_spinbox(
            100.0, 0.001, 1.0e7, 3, "mm"
        )
        self.fields["mesh_size_max_mm"] = make_double_spinbox(
            1000.0, 0.001, 1.0e7, 3, "mm"
        )
        self.fields["model_width_mm"] = make_double_spinbox(
            1000.0, 0.001, 1.0e9, 3, "mm"
        )
        mesh_form.addRow("Minimum mesh size", self.fields["mesh_size_min_mm"])
        mesh_form.addRow("Maximum mesh size", self.fields["mesh_size_max_mm"])
        mesh_form.addRow("Out-of-plane width", self.fields["model_width_mm"])
        layout.addWidget(mesh_group)

        note = QLabel(
            "The preview displays only the selected MSH file. Its exact node "
            "tags and coordinates are used for live load placement. The STEP "
            "path is used only when mesh regeneration is requested. The "
            "100–1000 mm defaults produce a practical trial mesh; reducing "
            "them can increase the element count dramatically."
        )
        note.setWordWrap(True)
        note.setObjectName("InformationLabel")
        layout.addWidget(note)

        self.step_selector.path_changed.connect(self._mark_mesh_preview_stale)
        self.mesh_selector.path_changed.connect(self._mark_mesh_preview_stale)
        self.recreate_mesh.toggled.connect(self._mesh_settings_changed)
        self.fields["mesh_size_min_mm"].valueChanged.connect(
            self._mark_mesh_preview_stale
        )
        self.fields["mesh_size_max_mm"].valueChanged.connect(
            self._mark_mesh_preview_stale
        )
        layout.addStretch(1)
        return self._scrollable_tab(content)

    def _build_soil_tab(self):
        content = QWidget()
        layout = QVBoxLayout(content)

        elastic_group = QGroupBox("Elastic and density properties")
        elastic_form = QFormLayout(elastic_group)
        self.fields["soil_e_kpa"] = make_double_spinbox(
            30_000.0, 0.001, 1.0e12, 2, "kPa"
        )
        self.fields["soil_poissons_ratio"] = make_double_spinbox(
            0.30, -0.99, 0.4999, 4, "", 0.01
        )
        self.fields["soil_density_mg_m3"] = make_double_spinbox(
            1.8, 0.0, 100.0, 4, "Mg/m³", 0.05
        )
        elastic_form.addRow("Young's modulus", self.fields["soil_e_kpa"])
        elastic_form.addRow(
            "Poisson's ratio",
            self.fields["soil_poissons_ratio"],
        )
        elastic_form.addRow("Density", self.fields["soil_density_mg_m3"])
        layout.addWidget(elastic_group)

        plastic_group = QGroupBox("Drucker–Prager properties")
        plastic_form = QFormLayout(plastic_group)
        self.fields["friction_angle_deg"] = make_double_spinbox(
            32.0, 0.0, 89.9, 3, "deg", 0.5
        )
        self.fields["cohesion_kpa"] = make_double_spinbox(
            1.0, 0.000001, 1.0e9, 4, "kPa", 0.1
        )
        self.fields["dilation_angle_deg"] = make_double_spinbox(
            0.0, 0.0, 89.9, 3, "deg", 0.5
        )
        self.fields["atmospheric_pressure_kpa"] = make_double_spinbox(
            101.0, 0.000001, 1.0e9, 3, "kPa"
        )
        self.fields["hardening_ratio"] = make_double_spinbox(
            1.0, 0.0, 1.0e6, 6, "H/G", 0.05
        )
        plastic_form.addRow(
            "Internal friction angle",
            self.fields["friction_angle_deg"],
        )
        plastic_form.addRow("Cohesion", self.fields["cohesion_kpa"])
        plastic_form.addRow(
            "Dilation angle",
            self.fields["dilation_angle_deg"],
        )
        plastic_form.addRow(
            "Atmospheric pressure",
            self.fields["atmospheric_pressure_kpa"],
        )
        plastic_form.addRow(
            "Hardening ratio",
            self.fields["hardening_ratio"],
        )
        layout.addWidget(plastic_group)

        self.fields["run_elastic_comparison"] = QCheckBox(
            "Run matching ElasticIsotropic comparison"
        )
        self.fields["run_elastic_comparison"].setChecked(True)
        layout.addWidget(self.fields["run_elastic_comparison"])

        warning = QLabel(
            "Hardening changes the physical post-yield response. H/G = 1.0 "
            "means the hardening modulus equals the elastic shear modulus."
        )
        warning.setWordWrap(True)
        warning.setObjectName("WarningLabel")
        layout.addWidget(warning)
        layout.addStretch(1)
        return self._scrollable_tab(content)

    def _build_beam_tab(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        beam_group = QGroupBox("Free-face beam")
        beam_form = QFormLayout(beam_group)
        self.fields["beam_e_n_mm2"] = make_double_spinbox(
            30_000.0, 0.001, 1.0e12, 2, "N/mm²"
        )
        self.fields["beam_thickness_mm"] = make_double_spinbox(
            300.0, 0.001, 1.0e9, 3, "mm"
        )
        self.fields["beam_width_mm"] = make_double_spinbox(
            1000.0, 0.001, 1.0e9, 3, "mm"
        )
        self.fields["coordinate_tolerance_mm"] = make_double_spinbox(
            1.0e-6, 1.0e-12, 1.0e3, 10, "mm"
        )
        beam_form.addRow("Young's modulus", self.fields["beam_e_n_mm2"])
        beam_form.addRow("In-plane thickness", self.fields["beam_thickness_mm"])
        beam_form.addRow("Out-of-plane width", self.fields["beam_width_mm"])
        beam_form.addRow(
            "Coordinate tolerance",
            self.fields["coordinate_tolerance_mm"],
        )
        layout.addWidget(beam_group)

        description = QLabel(
            "The beam follows the vertical free face beginning at (0, 0, 0). "
            "Its translational DOFs are bonded to the corresponding SSPquad "
            "soil nodes, and its origin is fixed in translation and rotation."
        )
        description.setWordWrap(True)
        description.setObjectName("InformationLabel")
        layout.addWidget(description)
        layout.addStretch(1)
        return self._scrollable_tab(content)

    def _build_loads_tab(self):
        content = QWidget()
        layout = QVBoxLayout(content)

        information = QLabel(
            "Loads redraw on the model immediately. Positive X acts right, "
            "positive Z acts upward, and positive moments are counter-clockwise."
        )
        information.setWordWrap(True)
        information.setObjectName("InformationLabel")
        layout.addWidget(information)

        load_tabs = QTabWidget()
        layout.addWidget(load_tabs, 1)

        point_page = QWidget()
        point_layout = QVBoxLayout(point_page)

        self.loads_table = QTableWidget(0, 6)
        self.loads_table.setHorizontalHeaderLabels(
            ["Target", "Node", "Magnitude", "Direction", "Units", ""]
        )
        self.loads_table.verticalHeader().setVisible(False)
        self.loads_table.setAlternatingRowColors(True)
        header = self.loads_table.horizontalHeader()
        header.setMinimumSectionSize(55)
        for column in range(6):
            header.setSectionResizeMode(
                column,
                QHeaderView.ResizeMode.Interactive,
            )
        for column, width in enumerate((82, 78, 155, 95, 58, 92)):
            self.loads_table.setColumnWidth(column, width)
        point_layout.addWidget(self.loads_table, 1)

        buttons = QHBoxLayout()
        add_button = QPushButton("Add Point Load")
        add_button.clicked.connect(lambda: self.add_load_row())
        buttons.addWidget(add_button)
        clear_button = QPushButton("Clear Loads")
        clear_button.clicked.connect(self.clear_load_rows)
        buttons.addWidget(clear_button)
        buttons.addStretch(1)
        point_layout.addLayout(buttons)
        load_tabs.addTab(point_page, "Point Loads")

        line_page = QWidget()
        line_layout = QVBoxLayout(line_page)
        line_note = QLabel(
            "Define uniform or trapezoidal boundary loading between two soil "
            "or beam nodes. Intensities use N/mm and redraw immediately."
        )
        line_note.setWordWrap(True)
        line_note.setObjectName("InformationLabel")
        line_layout.addWidget(line_note)

        self.line_loads_table = QTableWidget(0, 8)
        self.line_loads_table.setHorizontalHeaderLabels(
            [
                "Target",
                "Start node",
                "End node",
                "Start q",
                "End q",
                "Direction",
                "Units",
                "",
            ]
        )
        self.line_loads_table.verticalHeader().setVisible(False)
        self.line_loads_table.setAlternatingRowColors(True)
        line_header = self.line_loads_table.horizontalHeader()
        line_header.setMinimumSectionSize(60)
        for column in range(8):
            line_header.setSectionResizeMode(
                column,
                QHeaderView.ResizeMode.Interactive,
            )
        for column, width in enumerate((82, 96, 96, 130, 130, 82, 65, 92)):
            self.line_loads_table.setColumnWidth(column, width)
        line_layout.addWidget(self.line_loads_table, 1)

        line_buttons = QHBoxLayout()
        add_line_button = QPushButton("Add Line Load")
        add_line_button.clicked.connect(lambda: self.add_line_load_row())
        line_buttons.addWidget(add_line_button)
        clear_line_button = QPushButton("Clear Line Loads")
        clear_line_button.clicked.connect(self.clear_line_load_rows)
        line_buttons.addWidget(clear_line_button)
        line_buttons.addStretch(1)
        line_layout.addLayout(line_buttons)
        load_tabs.addTab(line_page, "Line Loads")
        return content

    def _build_analysis_tab(self):
        content = QWidget()
        layout = QVBoxLayout(content)

        analysis_group = QGroupBox("Static analysis")
        analysis_form = QFormLayout(analysis_group)
        self.fields["gravity_m_s2"] = make_double_spinbox(
            9.81, 0.0, 1.0e6, 5, "m/s²", 0.01
        )
        self.fields["gravity_steps"] = make_integer_spinbox(100)
        self.fields["point_load_steps"] = make_integer_spinbox(100)
        self.fields["convergence_tolerance"] = make_double_spinbox(
            1.0e-5, 1.0e-14, 1.0, 12
        )
        self.fields["maximum_iterations"] = make_integer_spinbox(100)
        analysis_form.addRow("Gravity acceleration", self.fields["gravity_m_s2"])
        analysis_form.addRow("Gravity increments", self.fields["gravity_steps"])
        analysis_form.addRow(
            "Point-load increments",
            self.fields["point_load_steps"],
        )
        analysis_form.addRow(
            "Convergence tolerance",
            self.fields["convergence_tolerance"],
        )
        analysis_form.addRow(
            "Maximum iterations",
            self.fields["maximum_iterations"],
        )
        layout.addWidget(analysis_group)

        convergence_group = QGroupBox("Live OpenSees convergence")
        convergence_layout = QVBoxLayout(convergence_group)
        convergence_note = QLabel(
            "Norm is the NormDispIncr displacement-increment norm for the "
            "current load step. A step converges when this is within the "
            "requested tolerance before the iteration limit is reached."
        )
        convergence_note.setWordWrap(True)
        convergence_note.setObjectName("InformationLabel")
        convergence_layout.addWidget(convergence_note)
        self.convergence_table = QTableWidget(0, 7)
        self.convergence_table.setHorizontalHeaderLabels(
            [
                "Case",
                "Phase",
                "Step",
                "Iterations",
                "Norm",
                "Tolerance",
                "Status",
            ]
        )
        self.convergence_table.verticalHeader().setVisible(False)
        self.convergence_table.setAlternatingRowColors(True)
        self.convergence_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        convergence_header = self.convergence_table.horizontalHeader()
        for column in range(7):
            convergence_header.setSectionResizeMode(
                column,
                QHeaderView.ResizeMode.ResizeToContents,
            )
        convergence_header.setSectionResizeMode(
            0,
            QHeaderView.ResizeMode.Stretch,
        )
        convergence_header.setSectionResizeMode(
            1,
            QHeaderView.ResizeMode.Stretch,
        )
        self.convergence_table.setMinimumHeight(175)
        convergence_layout.addWidget(self.convergence_table)
        layout.addWidget(convergence_group)

        output_group = QGroupBox("Results")
        output_form = QFormLayout(output_group)
        self.output_selector = PathSelector(
            "Select results folder",
            "",
            directory=True,
        )
        self.fields["output_directory"] = self.output_selector
        self.fields["save_plots"] = QCheckBox("Save result figures")
        self.fields["save_plots"].setChecked(True)
        self.fields["show_results_plot"] = QCheckBox(
            "Show result figures after analysis"
        )
        self.fields["show_results_plot"].setChecked(True)
        self.fields["show_node_tags_on_results"] = QCheckBox(
            "Show node tags on result plots"
        )
        self.fields["show_node_tags_on_results"].setChecked(True)
        output_form.addRow("Output folder", self.output_selector)
        output_form.addRow("", self.fields["save_plots"])
        output_form.addRow("", self.fields["show_results_plot"])
        output_form.addRow("", self.fields["show_node_tags_on_results"])
        layout.addWidget(output_group)

        analysis_note = QLabel(
            "Run Analysis creates or reads the mesh, applies gravity first, "
            "then applies all point and distributed line loads. The elastic "
            "comparison uses the identical mesh, beam, supports and loading."
        )
        analysis_note.setWordWrap(True)
        analysis_note.setObjectName("InformationLabel")
        layout.addWidget(analysis_note)
        layout.addStretch(1)
        return self._scrollable_tab(content)

    def _apply_style(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #f4f6f8;
                color: #1f2933;
                font-family: "Segoe UI";
                font-size: 10pt;
            }
            QFrame#HeaderFrame {
                background: #173f5f;
                border-radius: 7px;
            }
            QLabel#AppTitle {
                background: transparent;
                color: white;
                font-size: 19pt;
                font-weight: 700;
            }
            QLabel#AppSubtitle {
                background: transparent;
                color: #d8e8f3;
            }
            QLabel#InformationLabel {
                background: #eaf2f8;
                border: 1px solid #b8d2e3;
                border-radius: 5px;
                padding: 8px;
                color: #294d66;
            }
            QLabel#WarningLabel {
                background: #fff5dc;
                border: 1px solid #e5c36b;
                border-radius: 5px;
                padding: 8px;
                color: #6a4a00;
            }
            QGroupBox {
                background: white;
                border: 1px solid #cbd5df;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 8px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit,
            QTableWidget {
                background: white;
                border: 1px solid #b9c5d0;
                border-radius: 4px;
                padding: 4px;
                selection-background-color: #2878a8;
            }
            QPushButton {
                background: #2878a8;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 7px 12px;
                font-weight: 600;
            }
            QPushButton:hover { background: #1f6590; }
            QPushButton:pressed { background: #184e70; }
            QPushButton:disabled { background: #aab5bf; color: #e9edf0; }
            QTabWidget::pane {
                border: 1px solid #cbd5df;
                background: white;
            }
            QTabBar::tab {
                background: #dfe7ed;
                padding: 9px 12px;
                border: 1px solid #c3ced7;
            }
            QTabBar::tab:selected {
                background: white;
                border-bottom-color: white;
                font-weight: 600;
            }
            """
        )

    def _set_defaults(self):
        default_step = DEFAULT_PROJECT_DIR / "TIB BIN RAMP FEA 1.stp"
        default_mesh = DEFAULT_PROJECT_DIR / "TIB BIN RAMP FEA 1.msh"
        self.step_selector.setText(default_step)
        self.mesh_selector.setText(default_mesh)
        self.output_selector.setText(DEFAULT_PROJECT_DIR / "results")
        self.recreate_mesh.setChecked(False)
        self._update_render_button_text()

    def _connect_live_load_widget(self, widget):
        if isinstance(widget, QComboBox):
            widget.currentTextChanged.connect(self.schedule_live_redraw)
        elif isinstance(widget, QSpinBox):
            widget.valueChanged.connect(self.schedule_live_redraw)
        elif isinstance(widget, QDoubleSpinBox):
            widget.valueChanged.connect(self.schedule_live_redraw)

    def add_load_row(self, load=None):
        load = load or {
            "target": "Soil",
            "node_tag": 1,
            "magnitude": 0.0,
            "direction": "z",
        }
        row = self.loads_table.rowCount()
        self.loads_table.insertRow(row)

        target = QComboBox()
        target.addItems(["Soil", "Beam"])
        target.setCurrentText(str(load.get("target", "Soil")))

        node = QSpinBox()
        node.setRange(1, 2_000_000_000)
        node.setValue(int(load.get("node_tag", 1)))
        node.setKeyboardTracking(True)

        magnitude = QDoubleSpinBox()
        magnitude.setRange(-1.0e15, 1.0e15)
        magnitude.setDecimals(3)
        magnitude.setValue(float(load.get("magnitude", 0.0)))
        magnitude.setKeyboardTracking(True)
        magnitude.setGroupSeparatorShown(True)
        magnitude.setMinimumWidth(135)

        direction = QComboBox()
        direction.addItems(["x", "z", "moment"])
        direction.setCurrentText(str(load.get("direction", "z")))

        units = QLabel("N")
        units.setAlignment(Qt.AlignmentFlag.AlignCenter)

        remove = QPushButton("Remove")
        remove.setProperty("load_row", row)
        remove.clicked.connect(self.remove_load_row_from_button)

        widgets = {
            "target": target,
            "node": node,
            "magnitude": magnitude,
            "direction": direction,
            "units": units,
            "remove": remove,
        }
        self.load_rows.append(widgets)

        self.loads_table.setCellWidget(row, 0, target)
        self.loads_table.setCellWidget(row, 1, node)
        self.loads_table.setCellWidget(row, 2, magnitude)
        self.loads_table.setCellWidget(row, 3, direction)
        self.loads_table.setCellWidget(row, 4, units)
        self.loads_table.setCellWidget(row, 5, remove)

        for live_widget in (target, node, magnitude, direction):
            self._connect_live_load_widget(live_widget)
        direction.currentTextChanged.connect(
            lambda text, label=units: label.setText(
                "N.mm" if text == "moment" else "N"
            )
        )
        units.setText("N.mm" if direction.currentText() == "moment" else "N")
        self._refresh_remove_button_rows()
        self.schedule_live_redraw()

    def _refresh_remove_button_rows(self):
        for index, widgets in enumerate(self.load_rows):
            widgets["remove"].setProperty("load_row", index)

    @Slot()
    def remove_load_row_from_button(self):
        sender = self.sender()
        row = int(sender.property("load_row"))
        if 0 <= row < len(self.load_rows):
            self.loads_table.removeRow(row)
            self.load_rows.pop(row)
            self._refresh_remove_button_rows()
            self.schedule_live_redraw()

    @Slot()
    def clear_load_rows(self):
        self.loads_table.setRowCount(0)
        self.load_rows.clear()
        self.schedule_live_redraw()

    def collect_loads(self):
        return [
            {
                "target": widgets["target"].currentText(),
                "node_tag": widgets["node"].value(),
                "magnitude": widgets["magnitude"].value(),
                "direction": widgets["direction"].currentText(),
            }
            for widgets in self.load_rows
        ]

    def add_line_load_row(self, load=None):
        load = load or {
            "target": "Soil",
            "start_node": 1,
            "end_node": 2,
            "start_magnitude": 0.0,
            "end_magnitude": 0.0,
            "direction": "z",
        }
        row = self.line_loads_table.rowCount()
        self.line_loads_table.insertRow(row)

        target = QComboBox()
        target.addItems(["Soil", "Beam"])
        target.setCurrentText(str(load.get("target", "Soil")))

        start_node = QSpinBox()
        start_node.setRange(1, 2_000_000_000)
        start_node.setValue(int(load.get("start_node", 1)))
        start_node.setKeyboardTracking(True)

        end_node = QSpinBox()
        end_node.setRange(1, 2_000_000_000)
        end_node.setValue(int(load.get("end_node", 2)))
        end_node.setKeyboardTracking(True)

        start_magnitude = QDoubleSpinBox()
        start_magnitude.setRange(-1.0e12, 1.0e12)
        start_magnitude.setDecimals(4)
        start_magnitude.setValue(float(load.get("start_magnitude", 0.0)))
        start_magnitude.setKeyboardTracking(True)
        start_magnitude.setMinimumWidth(112)

        end_magnitude = QDoubleSpinBox()
        end_magnitude.setRange(-1.0e12, 1.0e12)
        end_magnitude.setDecimals(4)
        end_magnitude.setValue(float(load.get("end_magnitude", 0.0)))
        end_magnitude.setKeyboardTracking(True)
        end_magnitude.setMinimumWidth(112)

        direction = QComboBox()
        direction.addItems(["x", "z"])
        direction.setCurrentText(str(load.get("direction", "z")))

        units = QLabel("N/mm")
        units.setAlignment(Qt.AlignmentFlag.AlignCenter)
        remove = QPushButton("Remove")
        remove.setProperty("line_load_row", row)
        remove.clicked.connect(self.remove_line_load_row_from_button)

        widgets = {
            "target": target,
            "start_node": start_node,
            "end_node": end_node,
            "start_magnitude": start_magnitude,
            "end_magnitude": end_magnitude,
            "direction": direction,
            "units": units,
            "remove": remove,
        }
        self.line_load_rows.append(widgets)
        for column, key in enumerate(
            (
                "target",
                "start_node",
                "end_node",
                "start_magnitude",
                "end_magnitude",
                "direction",
                "units",
                "remove",
            )
        ):
            self.line_loads_table.setCellWidget(row, column, widgets[key])

        for live_widget in (
            target,
            start_node,
            end_node,
            start_magnitude,
            end_magnitude,
            direction,
        ):
            self._connect_live_load_widget(live_widget)
        self._refresh_line_load_button_rows()
        self.schedule_live_redraw()

    def _refresh_line_load_button_rows(self):
        for index, widgets in enumerate(self.line_load_rows):
            widgets["remove"].setProperty("line_load_row", index)

    @Slot()
    def remove_line_load_row_from_button(self):
        sender = self.sender()
        row = int(sender.property("line_load_row"))
        if 0 <= row < len(self.line_load_rows):
            self.line_loads_table.removeRow(row)
            self.line_load_rows.pop(row)
            self._refresh_line_load_button_rows()
            self.schedule_live_redraw()

    @Slot()
    def clear_line_load_rows(self):
        self.line_loads_table.setRowCount(0)
        self.line_load_rows.clear()
        self.schedule_live_redraw()

    def collect_line_loads(self):
        return [
            {
                "target": widgets["target"].currentText(),
                "start_node": widgets["start_node"].value(),
                "end_node": widgets["end_node"].value(),
                "start_magnitude": widgets["start_magnitude"].value(),
                "end_magnitude": widgets["end_magnitude"].value(),
                "direction": widgets["direction"].currentText(),
            }
            for widgets in self.line_load_rows
        ]

    @Slot()
    def schedule_live_redraw(self, *_):
        self.redraw_timer.start()

    @Slot()
    def update_live_load_preview(self):
        self.preview_canvas.set_loads(
            self.collect_loads(),
            self.collect_line_loads(),
        )

    def _update_render_button_text(self):
        if not hasattr(self, "render_button"):
            return
        self.render_button.setText(
            "Update && Render MSH"
            if self.recreate_mesh.isChecked()
            else "Reload MSH"
        )

    def _mark_mesh_preview_stale(self, *_):
        if hasattr(self, "preview_information"):
            self.preview_information.setText(
                "Mesh settings changed. Select Reload MSH, or enable mesh "
                "regeneration and select Update & Render MSH."
            )

    def _mesh_settings_changed(self, *_):
        self._update_render_button_text()
        self._mark_mesh_preview_stale()

    @Slot(bool)
    def preview_canvas_show_nodes(self, enabled):
        self.preview_canvas.set_show_node_tags(enabled)

    @Slot()
    def render_preview(self):
        self._start_mesh_preview(regenerate=self.recreate_mesh.isChecked())

    def reload_mesh_preview(self):
        self._start_mesh_preview(regenerate=False)

    def _start_mesh_preview(self, regenerate):
        if self.preview_thread is not None and self.preview_thread.isRunning():
            QMessageBox.information(
                self,
                "Preview in progress",
                "Please wait for the current geometry preview to finish.",
            )
            return

        step_path = self.step_selector.text()
        mesh_path = self.mesh_selector.text()
        if regenerate and not Path(step_path).is_file():
            QMessageBox.warning(self, "Missing STEP file", step_path)
            return
        if not regenerate and not Path(mesh_path).is_file():
            QMessageBox.warning(self, "Missing MSH file", mesh_path)
            return
        if (
            self.fields["mesh_size_min_mm"].value()
            > self.fields["mesh_size_max_mm"].value()
        ):
            QMessageBox.warning(
                self,
                "Invalid mesh sizes",
                "Minimum mesh size cannot exceed maximum mesh size.",
            )
            return

        self.render_button.setEnabled(False)
        action = "Updating and rendering" if regenerate else "Reloading"
        self.statusBar().showMessage(f"{action} MSH mesh…")
        self.log(f"{action} MSH mesh in the background…")

        self.preview_thread = QThread(self)
        self.preview_worker = GeometryPreviewWorker(
            mesh_path=mesh_path,
            step_path=step_path,
            regenerate=regenerate,
            mesh_size_min=self.fields["mesh_size_min_mm"].value(),
            mesh_size_max=self.fields["mesh_size_max_mm"].value(),
        )
        self.preview_worker.moveToThread(self.preview_thread)
        self.preview_thread.started.connect(self.preview_worker.run)
        self.preview_worker.completed.connect(self._preview_completed)
        self.preview_worker.failed.connect(self._preview_failed)
        self.preview_worker.completed.connect(self.preview_thread.quit)
        self.preview_worker.failed.connect(self.preview_thread.quit)
        self.preview_thread.finished.connect(self.preview_worker.deleteLater)
        self.preview_thread.finished.connect(self._preview_thread_finished)
        self.preview_thread.start()

    @Slot(object)
    def _preview_completed(self, geometry_data):
        self.geometry_data = geometry_data
        self.preview_canvas.set_geometry(geometry_data)
        self.preview_canvas.set_loads(
            self.collect_loads(),
            self.collect_line_loads(),
        )
        self.preview_information.setText(
            f"{'Updated and rendered' if geometry_data['regenerated'] else 'Rendered'} "
            f"{len(geometry_data['cells']):,} surface cells and "
            f"{len(geometry_data['nodes']):,} display nodes. Point-load "
            f"coordinates use the {geometry_data['node_source']} "
            f"({len(geometry_data['load_nodes']):,} nodes)."
        )
        self.log(self.preview_information.text())
        self.statusBar().showMessage("Geometry preview ready", 5000)

    @Slot(str)
    def _preview_failed(self, message):
        self.log(f"Preview failed: {message}")
        self.statusBar().showMessage("Preview failed", 5000)
        QMessageBox.critical(self, "Geometry preview failed", message)

    @Slot()
    def _preview_thread_finished(self):
        self.render_button.setEnabled(True)
        self._update_render_button_text()
        self.preview_thread = None
        self.preview_worker = None

    def _field_value(self, widget):
        if isinstance(widget, PathSelector):
            return widget.text()
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QSpinBox):
            return widget.value()
        if isinstance(widget, QDoubleSpinBox):
            return widget.value()
        if isinstance(widget, QComboBox):
            return widget.currentText()
        if isinstance(widget, QLineEdit):
            return widget.text()
        raise TypeError(f"Unsupported input widget: {type(widget).__name__}")

    def collect_configuration(self):
        values = {
            key: self._field_value(widget)
            for key, widget in self.fields.items()
        }
        return {
            "configuration_version": CONFIG_VERSION,
            "geometry": {
                key: values[key]
                for key in (
                    "step_path",
                    "mesh_path",
                    "recreate_mesh",
                    "mesh_size_min_mm",
                    "mesh_size_max_mm",
                    "model_width_mm",
                )
            },
            "soil": {
                key: values[key]
                for key in (
                    "soil_e_kpa",
                    "soil_poissons_ratio",
                    "soil_density_mg_m3",
                    "friction_angle_deg",
                    "cohesion_kpa",
                    "dilation_angle_deg",
                    "atmospheric_pressure_kpa",
                    "hardening_ratio",
                    "run_elastic_comparison",
                )
            },
            "beam": {
                key: values[key]
                for key in (
                    "beam_e_n_mm2",
                    "beam_thickness_mm",
                    "beam_width_mm",
                    "coordinate_tolerance_mm",
                )
            },
            "analysis": {
                key: values[key]
                for key in (
                    "gravity_m_s2",
                    "gravity_steps",
                    "point_load_steps",
                    "convergence_tolerance",
                    "maximum_iterations",
                )
            },
            "output": {
                key: values[key]
                for key in (
                    "output_directory",
                    "save_plots",
                    "show_results_plot",
                    "show_node_tags_on_results",
                )
            },
            "point_loads": self.collect_loads(),
            "line_loads": self.collect_line_loads(),
        }

    def _set_field_value(self, key, value):
        widget = self.fields.get(key)
        if widget is None:
            return
        if isinstance(widget, PathSelector):
            widget.setText(value)
        elif isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QSpinBox):
            widget.setValue(int(value))
        elif isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(value))
        elif isinstance(widget, QComboBox):
            widget.setCurrentText(str(value))
        elif isinstance(widget, QLineEdit):
            widget.setText(str(value))

    def apply_configuration(self, configuration):
        for section_name in ("geometry", "soil", "beam", "analysis", "output"):
            section = configuration.get(section_name, {})
            for key, value in section.items():
                self._set_field_value(key, value)

        self.clear_load_rows()
        for load in configuration.get("point_loads", []):
            self.add_load_row(load)
        self.clear_line_load_rows()
        for load in configuration.get("line_loads", []):
            self.add_line_load_row(load)
        self.schedule_live_redraw()

    @Slot()
    def save_configuration(self):
        suggested = DEFAULT_PROJECT_DIR / "retaining_wall_project.json"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save retaining-wall configuration",
            str(suggested),
            "JSON configuration (*.json)",
        )
        if not path:
            return
        Path(path).write_text(
            json.dumps(self.collect_configuration(), indent=2),
            encoding="utf-8",
        )
        self.log(f"Configuration saved: {path}")
        self.statusBar().showMessage("Configuration saved", 4000)

    @Slot()
    def load_configuration(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open retaining-wall configuration",
            str(DEFAULT_PROJECT_DIR),
            "JSON configuration (*.json)",
        )
        if not path:
            return
        try:
            configuration = json.loads(Path(path).read_text(encoding="utf-8"))
            self.apply_configuration(configuration)
        except Exception as exc:
            QMessageBox.critical(self, "Configuration error", str(exc))
            return
        self.log(f"Configuration loaded: {path}")
        self.statusBar().showMessage("Configuration loaded", 4000)

    @Slot()
    def reset_configuration(self):
        answer = QMessageBox.question(
            self,
            "Reset inputs",
            "Reset all inputs to their defaults?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._set_defaults()
        self.clear_load_rows()
        self.clear_line_load_rows()
        self.add_load_row(
            {
                "target": "Soil",
                "node_tag": 7,
                "magnitude": -100_000.0,
                "direction": "z",
            }
        )
        self.log("Inputs reset to defaults.")

    def _validation_messages(self):
        messages = []
        step_path = Path(self.step_selector.text())
        mesh_path = Path(self.mesh_selector.text())
        if self.recreate_mesh.isChecked() and not step_path.is_file():
            messages.append(f"STEP file does not exist: {step_path}")
        if not self.recreate_mesh.isChecked() and not mesh_path.is_file():
            messages.append(
                "MSH file must exist when mesh regeneration is disabled: "
                f"{mesh_path}"
            )

        minimum_size = self.fields["mesh_size_min_mm"].value()
        maximum_size = self.fields["mesh_size_max_mm"].value()
        if minimum_size > maximum_size:
            messages.append("Minimum mesh size cannot exceed maximum mesh size.")

        friction = self.fields["friction_angle_deg"].value()
        dilation = self.fields["dilation_angle_deg"].value()
        if dilation > friction:
            messages.append(
                "Dilation angle cannot exceed the internal friction angle."
            )

        poisson = self.fields["soil_poissons_ratio"].value()
        if not (-1.0 < poisson < 0.5):
            messages.append("Poisson's ratio must satisfy -1 < ν < 0.5.")

        if self.geometry_data is not None:
            load_nodes = self.geometry_data["load_nodes"]
            beam_map = ModelPreviewCanvas._beam_node_map(load_nodes)
            for index, load in enumerate(self.collect_loads(), start=1):
                node = load["node_tag"]
                if load["target"] == "Soil" and node not in load_nodes:
                    messages.append(
                        f"Load row {index}: soil node {node} is not in the mesh."
                    )
                if load["target"] == "Beam" and node not in beam_map:
                    messages.append(
                        f"Load row {index}: beam node {node} is not on the "
                        "predicted free-face beam."
                    )
                if load["target"] == "Soil" and load["direction"] == "moment":
                    messages.append(
                        f"Load row {index}: moments require a three-DOF beam node."
                    )

            for index, load in enumerate(self.collect_line_loads(), start=1):
                valid_nodes = (
                    load_nodes if load["target"] == "Soil" else beam_map
                )
                if load["start_node"] not in valid_nodes:
                    messages.append(
                        f"Line-load row {index}: start node "
                        f"{load['start_node']} is not a valid "
                        f"{load['target'].lower()} node."
                    )
                if load["end_node"] not in valid_nodes:
                    messages.append(
                        f"Line-load row {index}: end node "
                        f"{load['end_node']} is not a valid "
                        f"{load['target'].lower()} node."
                    )
                if load["start_node"] == load["end_node"]:
                    messages.append(
                        f"Line-load row {index}: start and end nodes must differ."
                    )
        return messages

    @Slot()
    def run_analysis(self):
        if self.analysis_thread is not None and self.analysis_thread.isRunning():
            QMessageBox.information(
                self,
                "Analysis in progress",
                "Please wait for the current OpenSees analysis to finish.",
            )
            return
        if self.preview_thread is not None and self.preview_thread.isRunning():
            QMessageBox.information(
                self,
                "Preview in progress",
                "Please wait for the geometry preview to finish.",
            )
            return

        messages = self._validation_messages()
        if messages:
            text = "\n".join(f"• {message}" for message in messages)
            self.log("Analysis not started because validation failed:\n" + text)
            QMessageBox.warning(self, "Cannot run analysis", text)
            return

        configuration = self.collect_configuration()
        solver_path = Path(__file__).resolve().with_name(
            "retaining_wall_model.py"
        )
        if not solver_path.is_file():
            QMessageBox.critical(
                self,
                "Missing solver",
                f"The analysis engine was not found:\n{solver_path}",
            )
            return

        self.run_button.setEnabled(False)
        self.render_button.setEnabled(False)
        self.validate_button.setEnabled(False)
        self.new_action.setEnabled(False)
        self.open_action.setEnabled(False)
        self.analysis_progress.setValue(0)
        self.analysis_progress.setFormat("Starting… %p%")
        self.analysis_progress.setVisible(True)
        self.statusBar().showMessage("Starting OpenSees analysis…")
        self.log("Starting complete analysis in the background…")
        self.convergence_table.setRowCount(0)
        self.convergence_row_keys.clear()
        self._last_convergence_log.clear()
        self._refresh_mesh_after_analysis = bool(
            configuration["geometry"]["recreate_mesh"]
        )
        self.tabs.setCurrentIndex(4)

        self.analysis_thread = QThread(self)
        self.analysis_worker = AnalysisWorker(configuration, solver_path)
        self.analysis_worker.moveToThread(self.analysis_thread)
        self.analysis_thread.started.connect(self.analysis_worker.run)
        self.analysis_worker.progress.connect(self._analysis_progress)
        self.analysis_worker.diagnostic.connect(self._analysis_diagnostic)
        self.analysis_worker.completed.connect(self._analysis_completed)
        self.analysis_worker.failed.connect(self._analysis_failed)
        self.analysis_worker.completed.connect(self.analysis_thread.quit)
        self.analysis_worker.failed.connect(self.analysis_thread.quit)
        self.analysis_thread.finished.connect(self.analysis_worker.deleteLater)
        self.analysis_thread.finished.connect(self._analysis_thread_finished)
        self.analysis_thread.start()

    @Slot(int, str)
    def _analysis_progress(self, value, message):
        self.analysis_progress.setValue(int(value))
        self.analysis_progress.setFormat(f"{message} — %p%")
        self.statusBar().showMessage(message)
        self.log(message)

    @Slot(object)
    def _analysis_diagnostic(self, diagnostic):
        case_name = str(diagnostic.get("case_name", "Analysis"))
        phase = str(diagnostic.get("phase", ""))
        key = (case_name, phase)
        if key not in self.convergence_row_keys:
            row = self.convergence_table.rowCount()
            self.convergence_table.insertRow(row)
            self.convergence_row_keys[key] = row
        row = self.convergence_row_keys[key]

        step = int(diagnostic.get("step", 0))
        total_steps = int(diagnostic.get("total_steps", 0))
        converged = bool(diagnostic.get("converged", False))
        return_code = int(diagnostic.get("return_code", 0))
        status = "Converged" if converged else f"FAILED ({return_code})"
        values = (
            case_name,
            phase,
            f"{step}/{total_steps}",
            str(int(diagnostic.get("iterations", 0))),
            (
                f"{float(diagnostic.get('convergence_norm')):.3e}"
                if math.isfinite(float(diagnostic.get("convergence_norm", math.nan)))
                else "—"
            ),
            f"{float(diagnostic.get('tolerance', math.nan)):.3e}",
            status,
        )
        for column, value in enumerate(values):
            item = self.convergence_table.item(row, column)
            if item is None:
                item = QTableWidgetItem()
                self.convergence_table.setItem(row, column, item)
            item.setText(value)
            if not converged:
                item.setBackground(QColor("#ffd9d5"))
            else:
                item.setBackground(QColor("#e4f4e7"))

        percent = int(diagnostic.get("overall_percent", 0))
        self.analysis_progress.setValue(percent)
        message = (
            f"{case_name} | {phase} {step}/{total_steps} | "
            f"iterations {diagnostic.get('iterations', 0)} | "
            f"norm {values[4]} | max displacement "
            f"{self._format_result_value(diagnostic.get('maximum_soil_displacement'))} mm"
        )
        self.analysis_progress.setFormat(f"{case_name}: {phase} — %p%")
        self.statusBar().showMessage(message)

        log_interval = max(total_steps // 10, 1)
        last_logged = self._last_convergence_log.get(key, 0)
        if not converged or step == 1 or step == total_steps or step - last_logged >= log_interval:
            self.log(message + f" | tolerance {values[5]} | {status}")
            self._last_convergence_log[key] = step

    @staticmethod
    def _format_result_value(value):
        if value is None:
            return "—"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not math.isfinite(numeric):
            return "—"
        return f"{numeric:.6g}"

    def _populate_results_table(self, result):
        records = list(result["comparison_records"])
        if not records:
            summary = result["summaries"][-1]
            records = [
                {
                    "Result": "Final maximum soil displacement",
                    "Unit": "mm",
                    "Drucker-Prager": summary[
                        "maximum_soil_displacement_mm"
                    ],
                },
                {
                    "Result": "Final horizontal support reaction",
                    "Unit": "N",
                    "Drucker-Prager": summary["horizontal_reaction_n"],
                },
                {
                    "Result": "Final vertical support reaction",
                    "Unit": "N",
                    "Drucker-Prager": summary["vertical_reaction_n"],
                },
                {
                    "Result": "Final beam-base moment",
                    "Unit": "N.mm",
                    "Drucker-Prager": summary["beam_base_moment_nmm"],
                },
                {
                    "Result": "Final maximum beam shear magnitude",
                    "Unit": "N",
                    "Drucker-Prager": summary["maximum_beam_shear_n"],
                },
                {
                    "Result": "Final maximum beam moment magnitude",
                    "Unit": "N.mm",
                    "Drucker-Prager": summary["maximum_beam_moment_nmm"],
                },
            ]

        keys = (
            "Result",
            "Unit",
            "Elastic",
            "Drucker-Prager",
            "Difference",
            "Difference (%)",
        )
        self.results_table.setRowCount(len(records))
        for row, record in enumerate(records):
            for column, key in enumerate(keys):
                value = record.get(key)
                text = str(value) if key in {"Result", "Unit"} and value else self._format_result_value(value)
                item = QTableWidgetItem(text)
                if column >= 2:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight
                        | Qt.AlignmentFlag.AlignVCenter
                    )
                self.results_table.setItem(row, column, item)

    @Slot(object)
    def _analysis_completed(self, result):
        self.last_analysis_result = result
        self._populate_results_table(result)
        self.results_canvas.set_results(result)

        summary_lines = []
        for summary in result["summaries"]:
            summary_lines.append(
                f"{summary['case_name']}: converged; maximum soil displacement "
                f"{summary['maximum_soil_displacement_mm']:.6g} mm at node "
                f"{summary['maximum_displacement_node']}; gravity equilibrium "
                f"error {summary['gravity_equilibrium_error_percent']:.4g}%."
            )
        self.result_status.setText("  ".join(summary_lines))

        output_directory = Path(result["output_directory"])
        if result["configuration"]["output"]["save_plots"]:
            figure_path = output_directory / "beam_force_comparison.png"
            self.results_canvas.figure.savefig(
                figure_path,
                dpi=300,
                bbox_inches="tight",
            )
            self.log(f"Result figure saved: {figure_path}")

        self.log(f"Analysis results saved in: {output_directory}")
        self.statusBar().showMessage("Analysis complete", 8000)
        if result["configuration"]["output"]["show_results_plot"]:
            self.output_tabs.setCurrentIndex(1)
        QMessageBox.information(
            self,
            "Analysis complete",
            "All requested cases converged. Results and tables have been "
            f"saved in:\n{output_directory}",
        )

    @Slot(str)
    def _analysis_failed(self, message):
        self.result_status.setText(
            "The most recent analysis did not complete. See the status log "
            "for the full OpenSees error."
        )
        self.log("Analysis failed:\n" + message)
        self.statusBar().showMessage("Analysis failed", 8000)
        diagnostic_text = message.strip()
        if "RuntimeError:" in diagnostic_text:
            diagnostic_text = diagnostic_text.rsplit("RuntimeError:", 1)[-1].strip()
        elif "ValueError:" in diagnostic_text:
            diagnostic_text = diagnostic_text.rsplit("ValueError:", 1)[-1].strip()
        QMessageBox.critical(
            self,
            "Analysis failed",
            "The OpenSees analysis did not complete. The full diagnostic "
            "has been written to the status log.\n\n"
            + diagnostic_text[-1800:],
        )

    @Slot()
    def _analysis_thread_finished(self):
        self.run_button.setEnabled(True)
        self.render_button.setEnabled(True)
        self.validate_button.setEnabled(True)
        self.new_action.setEnabled(True)
        self.open_action.setEnabled(True)
        self.analysis_progress.setVisible(False)
        self.analysis_thread = None
        self.analysis_worker = None
        if self._refresh_mesh_after_analysis:
            self._refresh_mesh_after_analysis = False
            QTimer.singleShot(0, self.reload_mesh_preview)

    @Slot()
    def validate_inputs(self):
        messages = self._validation_messages()
        if messages:
            text = "\n".join(f"• {message}" for message in messages)
            self.log("Input validation found issues:\n" + text)
            QMessageBox.warning(self, "Input validation", text)
            return False

        self.log("All currently available input checks passed.")
        QMessageBox.information(
            self,
            "Input validation",
            "All currently available input checks passed.",
        )
        return True

    def log(self, message):
        self.log_output.append(str(message))

    def closeEvent(self, event):
        if self.analysis_thread is not None and self.analysis_thread.isRunning():
            QMessageBox.warning(
                self,
                "Analysis in progress",
                "Wait for the current analysis to finish before closing the "
                "application.",
            )
            event.ignore()
            return
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("The Project Company")
    app.setFont(QFont("Segoe UI", 10))
    window = RetainingWallApp()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
