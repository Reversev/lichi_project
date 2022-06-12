g++ -o yolo-fastest yolo-fastest.cpp -I ../install/include/ncnn/ ../install/lib/libncnn.a `pkg-config --libs --cflags opencv4` -fopenmp
