## System Identification Framework ##

Localizes damage on structures by performing adjoint optimization over the Young modulus of the elements of a CalculiX model.

### Requirements ###

- [CalculiX](https://www.dhondt.de/)
- [PyROL](https://rol.sandia.gov/)
- [Python3](https://www.python.org/downloads/)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [vtk](https://docs.vtk.org/en/latest/api/python.html)

### Instalation ###

1. #### Install PyROL ####
   
   Download [PyROL](https://www.sandia.gov/app/uploads/sites/232/2024/12/pyrol-2024.9.13.13.29develop.4795e2b0.tar.gz), go to the download directory and run
   
   ```bash
   python3 -m pip install pyrol-2024.9.13.13.29develop.4795e2b0.tar.gz
   ```
   
3. #### Install CalculiX ####

   Download source code from [https://www.dhondt.de/](https://www.dhondt.de/) and follow the instalation guide. Alternative, you can directly download the Linux executable provided.
   
4. #### Link CalculiX ####

   This program will attempt to run CalculiX by using the command `ccx`. In order to do this you can create a symbolic link to your CalculiX executable.
   If, for instance, your CalculiX executable is located at `/home/user/Calculix/CalculiX/ccx_2.22/src/ccx_2.22_MT`, you can create a symbolic link to it by doing

   ```bash
   cd
   mkdir bin
   cd bin
   ln -s /home/user/Calculix/CalculiX/ccx_2.22/src/ccx_2.22_MT ccx
   ```
  
   Following this, make sure that this bin directory belongs to your PATH:

   ```bash
   cd
   nano .bashrc
   ```

   and add the following line:
   ```bash
   export PATH="/home/user/bin:$PATH"
   ```

   where `/home/user` should be replaced with your own home directory.

4. #### Install Remaining Libraries ####

   The remaining libraries can be installed using `pip` or `pip3`:

   ```bash
   pip install numpy scipy vtk
   ```

### Quick Start ###

Use the application present in the `apps` folder as reference.

### Peer Reviewed Articles ###

Below are the articles that make use of this framework. If you find them or this code useful, please make sure to cite them!

   1. Airaudo, Facundo N., et al. "[Adjoint-based determination of weaknesses in structures.](https://www.sciencedirect.com/science/article/abs/pii/S0045782523005959)" *Computer Methods in Applied Mechanics and Engineering* 417 (2023): 116471.

   2. Airaudo, Facundo N., Harbir Antil, and Rainald Löhner. "[Conditional value at risk for damage identification in structural digital twins.](https://www.sciencedirect.com/science/article/abs/pii/S0168874X25000058)" *Finite Elements in Analysis and Design* 245 (2025): 104316.

   3. Löhner, Rainald, et al. "[High‐fidelity digital twins: Detecting and localizing weaknesses in structures.](https://onlinelibrary.wiley.com/doi/full/10.1002/nme.7568)" *International Journal for Numerical Methods in Engineering* 125.21 (2024): e7568.

   4. Ansari, Talhah Shamshad Ali, et al. "[Adjoint-based recovery of thermal fields from displacement or strain measurements.](https://www.sciencedirect.com/science/article/pii/S0045782525000908)" *Computer Methods in Applied Mechanics and Engineering* 438 (2025): 117818.
