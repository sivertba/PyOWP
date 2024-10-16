# PyOWP

On-Water Processing module for Python

## Installation Guide

### Step 1: Install Conda

Conda is an open-source package management and environment management system that runs on Windows, macOS, and Linux. Follow the steps below to install Conda:

1. **Download the Conda Installer:**

   - Go to the [Miniconda download page](https://docs.conda.io/en/latest/miniconda.html).
   - Download the installer for your operating system.
2. **Run the Installer:**

   - **Windows:**
     - Double-click the `.exe` file you downloaded.
     - Follow the instructions in the setup wizard.
   - **macOS:**
     - Open the Terminal.
     - Run the following command:
       ```sh
       bash ~/Downloads/Miniconda3-latest-MacOSX-x86_64.sh
       ```
     - Follow the prompts on the installer screens.
   - **Linux:**
     - Open the Terminal.
     - Run the following command:
       ```sh
       bash ~/Downloads/Miniconda3-latest-Linux-x86_64.sh
       ```
     - Follow the prompts on the installer screens.
3. **Verify the Installation:**

   - Open a new terminal or command prompt.
   - Run the following command to verify that Conda is installed:
     ```sh
     conda --version
     ```

### Step 2: Create and Activate the Conda Environment

1. **Navigate to the Project Directory:**
   - Open a terminal or command prompt.
   - Navigate to the directory containing the

environment.yml

 file:
     ``sh      cd path/to/your/project      ``

2. **Create the Conda Environment:**
   - Run the following command to create the environment from the

environment.yml

 file:
     ``sh      conda env create -f environment.yml      ``

3. **Activate the Conda Environment:**
   - Run the following command to activate the environment:
     ```sh
     conda activate pyowp
     ```

### Step 3: Verify the Environment

1. **Check Installed Packages:**

   - Run the following command to list the installed packages and verify that the environment is set up correctly:
     ```sh
     conda list
     ```
2. **Deactivate the Environment:**

   - When you are done working in the environment, you can deactivate it by running:
     ```sh
     conda deactivate
     ```

### Additional Information

- **Updating the Environment:**
  - If you need to update the environment with new dependencies, modify the

environment.yml

 file and run:
    ``sh     conda env update -f environment.yml     ``

- **Removing the Environment:**
  - If you need to remove the environment, run:

    ```sh
    conda env remove -n pyowp
    ```

For more information on using Conda, refer to the [official Conda documentation](https://docs.conda.io/projects/conda/en/latest/index.html).

## Usage

With the pyowp enviroment activated open a new terminal and run the following to get helptext.

'''
> python onWaterRadiometry.py -h
'''

or the following to process data

'''
python OnWaterRadiometryProcessing.py -i exampleData.xlsx -o processedData.xlsx
'''

You have to organize your data as shown in 'exampleData.xlsx'