# FDSR-test
test version for FDSR
for searched small edsr in x4 and x2

## Dependencies
* Python 3.6
* PyTorch 1.5.1
* torchvision 0.6.1
* numpy
* matplotlib
* pillow

## 1. Download 
download main.py, searched_small_edsr folder
```bash
FDSR-test
|-- main.py
`-- searched_small_edsr_x4
    |-- model_best.pt
    |-- ..
`-- searched_small_edsr_x2
    |-- model_best.pt
    |-- ..

## 2. Place test image [.png]
you may delete testx4.png and place your own images

## 3. Execute
```bash
python main.py --scale 4 # for x4
python main.py --scale 2 # for x2
```
and then the output SR image will appear in the folder
