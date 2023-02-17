# batScripts
Python / Jupyter Lab Scripting for processing files from my AudioMoth for Bat ID

## Scripts

There are two scripts for running in Jupyter Lab or Notebook:

- [ExamineAudioMothFile.ipynb](ExamineAudioMothFile.ipynb)  : Look at an individual WAV file to examine the the spectrum / frequencies 
- ProcessAudioMothFiles.ipynb : Batch process a folder of WAV files. Files will be sorted into sub-folders by frequencies present.

Each of the above scripts has instructions on usage when opened in Jupyter.

## Running Scripts

To run these you need:

- Python : https://www.python.org/
- Jupyter Lab or Jupyter Notebook: https://jupyter.org/

I use Jupyter Lab. Jupyter is an interactive development environment for code and data based on Python.

I found the best way to run this is using a platform called Anaconda, a data science platform supporting various languages including Python. Anaconda allows to set up an environment with Jupyter and all the required Python packages etc so that the main OS environment doesn't have to be polluted with all of this. I use this on MacOS.

## Setup

Setting up can be tricky depending on the OS so some perseverence may be necessary.

Here is the rough guide to my setup if taking my scripts to use yourself - YMMV:

- Clone this repo from Github to your local HDD
- Install Anaconda free version from here: https://www.anaconda.com/
- Run up the Anaconda Navigator App
- Anaconda supports mutliple environments for langauges / package installation
- You can use the default, but I created a new one "BatCave" from Environments Section > "Create" button
- If creating a new environment, choose Python and the default version
- In the Anaconda Environment, install Jupyter Lab (it may be installed by default)
- I use VSCode (Visual Studio Code - free from Microsoft) for editing, so install that too if wanted
- Then you need to install various Anaconda and Python packages - see below.
- To run a script, launch Jupyter Lab in Anaconda. It will fire up in the web browser
- In Jupyter, browse to one of the ".ipynb" scripts and run 
- If the scripts fail, it is probably due to missing Python packages - see below
- Scripts can also be run from VSCode. It will connect to the Jupyter/Anaconda server

## Python Packages

To install packages, I used Python "pip" from the Terminal / bash.

- In Anaconda go to Enironments, click on your environment then "Open Terminal"
- This fires up a Terminal then do "pip/conda install" as per the list below

Here are the Anaconda packahes I installed first (I think this is the only one needed):

```
conda install -c conda-forge ipyfilechooser
```

Here are the Python packages I installed using 'pip':

```
pip install tinytag
pip install matplotlib.pyplot
pip install matplotlib
pip install scipy
pip install soundfile
pip install pydub
pip install IPython
pip install time
pip install pandas
pip install os
pip install fnmatch
pip install ipywidgets
```

## Troubleshooting

I try to keep this repo in a working state as is.

Usual errors are an "import" statement failing to find a Python package. If an import fails, try to find the name of the missing package and install it using a pip or conda install as shown in the previous list. It maybe one of these packages didn't install.

Apart from that, Google the error may find a solution to the issue.

These scripts use a File Choose UI to browse for a file / folder and that may be the issue. You could try coding the path directly in the script. The File Chooser is from the ipywidgets package - the latest version of this package is not compatible with running scripts in VSCode. For that to work, I had to downgrade the version using "pip install ipywidgets==7.7.2" to get ipywidgets version 7.7.2. This may get resolved after the time of writing this..

When running the scripts, each code section needs to be run. There is usually a setup section (to choose files, provide settings etc) then a processing section. Scripts may fail because previous sections have not run or all sections ran with defaults and did not find the intended WAV files because it is hard coded to my system. I might fix this with a config file in future, but it does not seem to be the interactive "Jupyter" way of doing things...

If the Jupyter notebook system is found to be annoying, just convert it all to straight Python code. Most of the heavy lifting is in the utilities.py package. Hard-core Pythonists may prefer it that way.

When processing a large batch of files, the output of images to the Notebook display may use a lot of system memory and choke up the system / OS. I'm looking to fix that with an option to just output spectra etc to image files.





