# YoloX Model for Patient Detection

# Install

Create new conda environment: 

``` shell
conda create -n yolox python=3.8
```

Download and install YOLOX in a separate folder:

``` shell
git clone git@github.com:Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip3 install -U pip && pip3 install -r requirements.txt
pip3 install -v -e .
```

Clone this repository: 

``` shell
git clone git@ssh.dev.azure.com:v3/dss-ml/toms-patient-detect/yolox
# or 
git clone https://dss-ml@dev.azure.com/dss-ml/toms-patient-detect/_git/yolox
cd yolox
```

Install dependencies:

``` shell
conda install -f environment.yml
```


# Usage

## Training

For training we use the `run_train.py` script that only takes 1 or two arguments:

```shell
python3 run_train.py <compute-instance> [-config <path-to-config-yaml-file>]
```

The `compute-instance` argument is a GPU instance on Azure or if `local` is specified, then training is executed on the local machine.
Details on deployment to Azure can be found in the code `src/azure_access.py`.


This will deploy the code to Azure ML. Details on how it is deployed can be found in the file `src/azure_access.py`.
The `-config` argument is optional. As default the file `configs/train_default.yml` is read.

There are more templates for training in the `configs/` folder. 

**Important** 

When experimenting and testing, use the `-config` argument freely. 
However, for serious and trackable development, follow this procedure:

  1. Edit the `configs/train_default.yml` file
  2. Make a commit
  3. Execute `run_train` without `-config`

This allows to track and reproduce the run easily including code and configuration. 
In Azure ML we would take note of the git repo, branch, and commit and could rerun the exact same experiment.


## Inference

By Inference we mean the prediction of objects in an image or video without calculating any scores. 
Only the bounding-box and confidence are returned and drawn onto the images that are passed to the model.

To execute inference use a config file in the same way as with training: 

``` shell
python3 run_infer.py <compute-target> [-config <path-to-config-yaml-file>]
```

If the argument `-config` is ommited, the default `configs/infer_default.yml` will be used.
To run inference on the local machine, use `local` for the `<compute-target>`.

The images that are created will be saved into the folder specified by `outdir`, defined in the configuration file.


## Evaluation 

Evaluation is similar to 'Inference', but with the difference that the predicted objects are compared to the ground-truth and the average precision is calculated. No bounding boxes are created, only the score is returned.

To execute evaluation use a config file in the same way as with training: 

``` shell
python3 run_eval.py <compute-target> [-config <path-to-config-yaml-file>]
```

If the argument `-config` is ommited, the default `configs/eval_default.yml` will be used.
To run evaluation on the local machine, use `local` for the `<compute-target>`.

A single log file will be created inside the folder specified by `outdir`, defined in the configuration file.

