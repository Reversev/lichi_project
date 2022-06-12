c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` LitchiLocation.cpp -o LitchiLocation`python3-config --extension-suffix`  -I ../install/include/ncnn/ ../install/lib/libncnn.a `pkg-config --libs --cflags opencv4` -fopenmp

