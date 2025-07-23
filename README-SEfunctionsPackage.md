SEfunctions
A Python package for simulating and analyzing spectroscopic ellipsometry and optical properties using advanced oscillator models.
Author: [Nandan, Alex]

**Includes:** Bruggeman EMA, Lorentz, Drude, Sellmeier oscillators, and SE simulation utilities.
-----------------------------------------------------------------------------------------------------
📦 Installation Steps (from ZIP)
Go to the repository:
https://github.com/nandan9989/mlRTSE

Download the ZIP file:

---Locate SEFunctions.zip in the repository files.

---Click on it, then use the “Download” button to save it to your computer.

Extract the ZIP:

---Find SEFunctions.zip in your Downloads (or the location you saved).

---Right-click and select Extract All (Windows) or unzip (Mac/Linux).

---You will get a folder called SEFunctions.

(Optional but recommended) Open a terminal and go into the folder:

---Open a terminal (Command Prompt/PowerShell on Windows, Terminal on Mac/Linux).

Change directory to where you extracted:

---cd path/to/SEFunctions

Install the package locally:

---Make sure you have Python and pip installed.

In the SEFunctions directory (the one with setup.py), run:

---pip install -e .

This makes the SEfunctions package available to any Python script on your system!

Ready to Use!
Now, in any Python file or notebook, you can import and use the functions:
from SEfunctions.core import Bruggeman_EMA_Roussel, Snells_Law, fresnel_coefficients
# ...and other functions
-----------------------------------------------------------------------------------------------------
🚀 Usage:
Import any function you need:

from SEfunctions.core import (
    Bruggeman_EMA_Roussel,
    Snells_Law,
    fresnel_coefficients,
    Scattering_Matrix,
    SE_Sim,
    Lorentz,
    drude_epsilon,
    Sellmeier,
    sumosscilator
)

-----------------------------------------------------------------------------------------------------
Example 1: Calculate Bruggeman EMA

# Suppose M1 and M2 are DataFrames for bulk and void with 'Wavelength (nm)' and 'N' columns
ema_df = Bruggeman_EMA_Roussel(M1, M2, c=0.3)
print(ema_df.head())


Example 2: Run an SE Simulation

layers = [layer1, layer2, layer3]    # List of DataFrames (with 'N' column)
AOI = 70.0                           # Angle of incidence in degrees
d = [100.0, 1.75, 200.0]             # Layer thicknesses in nm
result = SE_Sim(layers, AOI, d, NCS=True)
print(result.head())
-----------------------------------------------------------------------------------------------------

📖 Function Descriptions:
-----------------------------------------------------------------------------------------------------
Bruggeman_EMA_Roussel(M1, M2, c)
Purpose: Calculates the Bruggeman Effective Medium Approximation (EMA) between two materials M1 and M2 at fraction c of M1.

Inputs: 
M1, M2: Pandas DataFrames with columns 'Wavelength (nm)', 'N'
c: Fraction (0–1) of M1 in mixture

Returns: DataFrame of effective index N over wavelengths

-----------------------------------------------------------------------------------------------------
Snells_Law(Structure, AOI)
Purpose: Calculates refracted angles for each interface in a multilayer stack using Snell's Law.
Inputs:

Structure: List of DataFrames (must include 'N' column)

AOI: Angle of incidence in degrees
Returns: Array of angles for each layer and wavelength
-----------------------------------------------------------------------------------------------------
fresnel_coefficients(N, angles)
Purpose: Calculates the Fresnel reflection and transmission coefficients for s and p polarizations.
Inputs:

N: Array of refractive indices for each layer

angles: Array of incident angles
Returns: (rs, rp, ts, tp) — reflection & transmission coefficients
-----------------------------------------------------------------------------------------------------
Scattering_Matrix(N, angles, d, lam, r, t)
Purpose: Builds the total scattering matrix for the multilayer structure using transfer matrix method.
Inputs:

N, angles: Layer indices and angles

d: Thicknesses (nm)

lam: Wavelength array (nm)

r, t: Fresnel coefficients
Returns: Scattering matrix S
-----------------------------------------------------------------------------------------------------
SE_Sim(Structure, AOI, d, write_data=False, NCS=True)
Purpose: Simulates spectroscopic ellipsometry response for a multilayer stack.
Inputs:

Structure: List of DataFrames (with 'N' columns)

AOI: Angle of incidence

d: List/array of layer thicknesses

NCS: If True, returns N, C, S (else Psi/Delta)
Returns: DataFrame with spectra
-----------------------------------------------------------------------------------------------------
Lorentz(E, An, Br, En, einf=1.0)
Purpose: Generates dielectric function from Lorentz oscillator model.
Inputs:

E: Photon energies (eV)

An: Oscillator amplitude

Br: Broadening

En: Resonance energy (eV)

einf: High-frequency dielectric constant
Returns: DataFrame with real/imag parts and metadata
-----------------------------------------------------------------------------------------------------
drude_epsilon(E, rho_n, tau_fs)
Purpose: Generates dielectric function from Drude model for free carriers.
Inputs:

E: Photon energies (eV)

rho_n: Resistivity (Ohm·cm)

tau_fs: Scattering time (fs)
Returns: DataFrame
-----------------------------------------------------------------------------------------------------
Sellmeier(A_uv, A_ir, En, e_inf, E)
Purpose: Computes dielectric function using the Sellmeier model.
Inputs:

A_uv, A_ir: Absorption strengths

En: Resonance energy

e_inf: High-frequency dielectric

E: Energies
Returns: DataFrame
-----------------------------------------------------------------------------------------------------
sumosscilator(df_list)
Purpose: Sums the dielectric response from multiple oscillator DataFrames into a single composite dielectric function.
Inputs:

df_list: List of oscillator DataFrames (from above)
Returns: DataFrame with combined dielectric
-----------------------------------------------------------------------------------------------------

## 📝 Requirements

This package requires the following Python libraries:

- numpy
- pandas

**Optional (for certain tasks):**
- matplotlib (for plotting or visualization)
- tqdm (for progress bars)
- scipy (if you later add advanced numerical or fitting routines)

**Install requirements with:**
```bash
pip install numpy pandas matplotlib tqdm
-----------------------------------------------------------------------------------------------------

✏️ How to Edit or Add Functions in Your SEfunctions Package:

1. Find the Core Code: 
---Go to the folder where you extracted the SEFunctions zip.
---Open the folder named SEfunctions (note: this is inside the outer folder).

The main code is in the file:
SEfunctions/core.py

2. To Edit an Existing Function
---Open core.py in your favorite code editor (VS Code, Sublime, Notepad++, etc.).
---Find the function you want to change.
---Make your edits (change code, fix bugs, improve logic, etc.).

Save the file.

3. To Add a New Function
---In core.py, scroll to the end (or wherever you want).
---Define your new function as usual, e.g.:

def new_cool_function(arg1, arg2):
    # Your code here
    return something
Save the file.

4. (Optional but Recommended) Add Your New Function to __init__.py
---Open SEfunctions/__init__.py (it may be empty or already have imports).

Import your new function, for example:
from .core import Bruggeman_EMA_Roussel, Snells_Law, new_cool_function

5. Reinstall the Package (if installed in "editable" mode, just restart Python)
---If you previously installed with pip install -e ., just restart your Python script or notebook.

If not, re-run the install command:
pip install -e .

6. Now Import and Use the Updated/Added Functions.

*******************************************************************************************************************************************************************************************