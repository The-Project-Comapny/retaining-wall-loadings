# TPC Retaining Wall Soil Analysis

Author: Mishal Mohanlal  
Version: 1.0.0

This PySide6 application provides the complete interface and OpenSeesPy
workflow for the retaining-wall soil model. It uses SSPquad plane-strain soil
elements with a nonlinear Drucker-Prager material and a bonded free-face beam.
The application uses N, mm and seconds throughout.

## Install and start

Run `install_retaining_wall_gui.bat` once. It creates the dedicated
`retaining_wall_gui` Conda environment and installs PySide6, Gmsh,
Matplotlib, pandas, NumPy, OpenSeesPy and the cryptography dependency used by
licensing.

Start the application with `start_retaining_wall_app.bat`.

## Build the Windows executable

Run `build_retaining_wall_exe.bat`. It installs the build-only PyInstaller
dependency and creates:

```text
dist\TPC_Retaining_Wall_Analysis\TPC_Retaining_Wall_Analysis.exe
```

Distribute the complete `TPC_Retaining_Wall_Analysis` folder, not the EXE by
itself. The application is intentionally built in one-folder mode because the
OpenSees, Gmsh, Qt and plotting runtimes contain supporting DLLs and plugins.
The build includes the public license-verification key but never the private
administrator signing key.

## License activation

The application uses an offline, machine-bound signed license:

1. On first start, copy the activation request displayed by the application.
2. Send it to the license administrator.
3. The administrator runs `start_admin_license_generator.bat`, pastes the
   activation request, enters the licensed user and expiry date, and generates
   a license code.
4. Paste the returned license code into the activation window and select
   **Activate**.

The activated license is stored at:

```text
%LOCALAPPDATA%\TheProjectCompany\RetainingWallSoilAnalysis\license.json
```

`license_private_key.json` is an administrator secret. Keep it outside every
user distribution and never bundle it into the executable. User distributions
need only `license_public_key.json`.

## Workflow

1. Select the STEP and MSH paths under **Geometry & Mesh**.
2. Select **Reload MSH** to display the existing mesh. To regenerate it,
   enable regeneration, enter the mesh sizes, and select **Update & Render
   MSH**. Only the MSH mesh is displayed.
3. Enter the soil stiffness, density, Drucker-Prager, beam and analysis
   properties.
4. Add point loads or uniform/trapezoidal line loads. Loads redraw immediately.
5. Select **Run Analysis** and follow the live OpenSees convergence data.
6. Review the embedded shear and bending-moment diagrams under **Results**.

Gravity is solved and held first. Point and distributed loads are then applied
incrementally to the nonlinear model.

## Load behaviour

- Point forces accept global X or Z in N.
- Point moments accept N.mm and require a three-DOF beam node.
- Soil line loads use N/mm and are converted to consistent nodal forces along
  the shortest connected exterior SSPquad boundary path.
- Beam line loads use OpenSees beam element loads on every selected free-face
  member. Trapezoidal loading uses the exact average intensity on each member.

## Saved results

Each successful run writes:

- `analysis_configuration.json`
- `analysis_summary.csv`
- `analysis_results.txt`
- `convergence_history.csv`
- `beam_actions.png` when figure saving is enabled

The text report begins with the plain heading `Results` and includes the
author, application version, material model, units and numerical results. The
status log contains progress messages and the complete diagnostic if an
analysis does not converge.
