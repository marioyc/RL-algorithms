apt-get update
apt-get upgrade

apt-get install -y python2.7-dev python-pip

pip install numpy

sudo apt-get install -y libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev cmake
git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git ~/ALE
cd ~/ALE
mkdir build
cd build
cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON ..
make -j 4
pip install ..
