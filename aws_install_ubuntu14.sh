mkdir logs
mkdir stats
mkdir video
mkdir weights

apt-get update
apt-get upgrade

apt-get install -y python2.7-dev python-pip

pip install numpy

# install ALE
sudo apt-get install -y libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev cmake
git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git ~/ALE
cd ~/ALE
mkdir build
cd build
cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON ..
make -j 4
pip install ..

# install OpenCV
git clone https://github.com/Itseez/opencv.git ~/opencv
cd ~/opencv
git checkout 3.1.0
mkdir build
cd build
cmake .. -DBUILD_opencv_python2=ON
make -j 4
make install

# install joblib
pip install joblib
