# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_add_python_bindings_test(
  "Unit.Ylm.Python.Strahlkorper"
  Test_Strahlkorper.py
  "Unit;Python"
  PySphericalHarmonics)

  spectre_add_python_bindings_test(
  "Unit.Ylm.Python.Spherepack"
  Test_Spherepack.py
  "Unit;Python"
  PySphericalHarmonics)
