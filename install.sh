# Clone Few Shots Vid2Vid (Nvidia) repo and install it
git clone https://github.com/NVlabs/few-shot-vid2vid.git

# Clone MaskRCNN Benchmark repo and install it
git clone https://github.com/facebookresearch/maskrcnn-benchmark.git 
cd maskrcnn-Benchmark && pip install .
cd ..

# Create conda env and install requirements
conda env create  --file requirements.txt

#Download datasets
#wget...