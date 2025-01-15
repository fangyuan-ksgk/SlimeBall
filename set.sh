# Check if environment exists and activate, or create if it doesn't
if conda env list | grep -q "volleyball"; then
    conda activate volleyball
else
    conda create -n volleyball python=3.9 -y
    conda activate volleyball
fi


# Install dependencies
pip install setuptools==65.5.0 pip==21
pip install gym==0.19.0
pip install pyglet==1.5.27
pip install "opencv-python<=4.3"
pip uninstall numpy
pip install "numpy<2.0"
pip install pygame==2.6.1
pip install graphviz
pip install neat-python
pip install matplotlib
pip install mpi4py 
brew install open-mpi
pip install ipywidgets

pip install seaborn
