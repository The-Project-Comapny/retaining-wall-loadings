# Retaining Wall PySide6 Analysis System

This application provides the complete interface and OpenSeesPy analysis
workflow for the retaining-wall soil model. The model uses SSPquad plane-strain
soil elements, an elastic beam bonded to the vertical free face, and a choice
of ElasticIsotropic or Drucker-Prager soil response.

## Install

Double-click `install_retaining_wall_gui.bat` once. It creates the clean
`retaining_wall_gui` Conda environment and installs PySide6, Gmsh,
Matplotlib, pandas, NumPy and OpenSeesPy.

The environment is deliberately separate from `opensees_final_build`, whose
PyQt installation conflicts with the PySide6 Qt runtime.

## Start

Double-click `start_retaining_wall_app.bat`.

The application executable is:

```text
%USERPROFILE%\miniconda3\envs\retaining_wall_gui\python.exe
```

and it launches `retaining_wall_app.py` from this folder.

## Workflow

1. Select the STEP and MSH paths under **Geometry & Mesh**.
2. Select **Reload MSH** to display the existing mesh. To change the mesh,
   enable regeneration, enter the mesh sizes, and select **Update & Render
   MSH**. The STEP geometry is never displayed.
3. Enter the soil, beam and analysis properties. All analysis units are N, mm
   and seconds; the interface converts kPa and Mg/m³ automatically.
4. Add point loads or uniform/trapezoidal line loads. Loads redraw immediately.
5. Select **Run Analysis**.
6. Follow the live OpenSees iteration count and NormDispIncr convergence norm
   under **Analysis & Output**.
7. Review the side-by-side result table and embedded shear/moment diagrams in
   the **Results** tab.

Gravity is solved and held first. Point and distributed loads are then applied
incrementally. If elastic comparison is enabled, the ElasticIsotropic and
Drucker-Prager cases use identical geometry, mesh, beam, supports and loads.

## Load behaviour

- Point forces accept global X or Z in N.
- Point moments accept N.mm and require a three-DOF beam node.
- Soil line loads use N/mm and are converted to consistent nodal forces along
  the shortest connected exterior SSPquad boundary path.
- Beam line loads use OpenSees beam element loads on every selected free-face
  member. Trapezoidal loading is represented by the exact average intensity on
  each member.

## Saved results

Each successful run writes the following files to the selected results folder:

- `analysis_configuration.json`
- `analysis_summary.csv`
- `convergence_history.csv`
- `elastic_vs_drucker_prager.csv` when comparison is enabled
- `beam_force_comparison.png` when figure saving is enabled

The status log contains progress messages and the complete diagnostic if an
analysis does not converge.
