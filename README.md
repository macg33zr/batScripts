# batScripts
Python / Jupyter Lab Scripting for processing files from my AudioMoth for Bat ID

## Scripts

There are two scripts for running in Jupyter Lab or Notebook:

- ExamineAudioMothFile.ipynb  : Look at an individual WAV file and look at the spectrum / frequencies 
- ProcessAudioMothFiles.ipynb : Batch process a folder of WAV files. Files will be sorted into sub-folders by frequencies present.

Each of the above scripts has instructions on usage when opened in Jupyter.

## Running Scripts

To run these you need:

- Python : https://www.python.org/
- Jupyter Lab or Jupyter Notebook: https://jupyter.org/

I use Jupyter Lab. Jupyter is an interactive development environment for code and data based on Python.

I found the best way to run this is using a platform called Anaconda, a data science platform supporting various languages including Python. Anaconda allows to set up an environment with Jupyter and all the required Python packages etc so that the main OS environment doesn't have to be polluted with all of this. I use this on MacOS.

## Setup

Setting up can be tricky depending on the OS so some peserverence may be necessary.

Here is the rough guide to my setup if taking my scripts to use yourself YMMV:

- Checkout this repo from Github to your local HDD
- Install Anaconda free version from here: https://www.anaconda.com/
- Run up the Anaconda Navigator App
- Anaconda supports mutliple environments for langauges / package installation
- You can use the default, but I created a new one "BatCave" in Envronments Section > "Create" button
- In the Anaconda Environment, install Jupyter Lab (it may be installed by default)
- I use VSCode (Visual Studio Code - free from Microsoft) for editing, so install that too if wanted
- Then you need to install varioys Python packages - see below.
- To run a script, launch Jupyter Lab in Anaconda. It will fire up in the web browser
- In Jupyter, browse to one of the ".ipynb" scripts and run 
- If the scripts fail, it is probably due to missing Python packages - see below

## Python Packages

To install packages, I used Python "pip" from the Terminal / bash.

- In Anaconda go to Enironments, click on your environment then "Open Terminal"
- This fires up a Terminal then do "pip install" as per the list below

Here are the packages I installed:

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





