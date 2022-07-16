FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu18.04 AS base

FROM base AS torch

RUN pip install 'numpy>=1.20' \
    'pip>=21.2.1'
RUN conda install pytorch torchvision torchaudio cudatoolkit=11.1 ruamel.yaml -c pytorch -c nvidia

FROM torch AS pip
# Install pip dependencies
RUN pip install 'matplotlib>=3.3' \
                'psutil>=5.8' \
                'tqdm>=4.59' \
                'pandas>=1.1' \
                'scipy>=1.5' \
                'azureml-core==1.33.0.post1' \
                'azureml-defaults==1.33.0' \
                'azureml-mlflow==1.33.0' \
                'azureml-telemetry==1.33.0' \
                'onnxruntime-gpu>=1.7' \
                'opencv_python' \
                'scikit-image' \
                'Pillow' \
                'thop' \
                'cython' \
                'onnx==1.8.1' \
                'onnxruntime==1.8.0' \
                'onnx-simplifier==0.3.5' \
                'tabulate' \
                'ninja' \
                'loguru' \
                'tensorboard'

FROM pip AS pycocotools
RUN pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

FROM pycocotools AS yolox

# yolox version is pinned here
RUN git clone -b 0.2.0 --single-branch https://github.com/Megvii-BaseDetection/YOLOX.git
RUN cd YOLOX && pip3 install -v -e .

FROM  yolox AS libs
# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH

RUN apt-get --assume-yes install libglu1-mesa

FROM libs AS fiftyone
RUN pip install 'fiftyone'

FROM fiftyone AS addlib
RUN pip install 'ruamel.yaml'



