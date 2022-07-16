# Document Title

* General note

This folder contains templates that can be used to define training, inference, or evaluation tasks. 

For experimentation, configs from this folder can be directly used such as this: 

```
./run_train.py <compute_instance> -config config_templates/train__<config-name>.yml
./run_eval.py <compute_instance> -config config_templates/eval__<config-name>.yml
./run_infer.py <compute_instance> -config config_templates/infer__<config-name>.yml
```

The configuration file can of course be located in another place. 

For serious experimentation runs that need to be tracked, you need to first modify the default config files:

 - for training: `configs/train_default.yml`
 - for evaluation: `configs/eval_default.yml`
 - for inference: `configs/infer_default.yml`
 
In this case, ommit the config file when starting a run: 
 
 ```
./run_train.py <compute_instance> 
./run_eval.py <compute_instance>
./run_infer.py <compute_instance>
```

**Make sure to commit the git repo before executing the code so it is saved and can be revised and rerun.**
