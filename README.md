# Welcome to FastFlow
Shield: [![CC BY-NC-ND 4.0][cc-by-shield]][cc-by-nc-nd] [![CC BY-NC-ND 4.0][cc-by-nc-nd-image]][cc-by-nc-nd]

[cc-by-nc-nd]: http://creativecommons.org/licenses/by-nc-nd/4.0/
[cc-by-nc-nd-image]: https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png
[cc-by-nc-nd-shield]: https://upload.wikimedia.org/wikipedia/commons/7/73/Cc_by-nc-nd_icon.svg


FastFlow is a deep learning training system that automatically detects CPU bottlenecks in deep learning training pipelines and resolves the bottlenecks with data pipeline offloading to remote resources.
We currently implement FastFlow on top of TensorFlow 2.7.0.


## Install
To use FastFlow, you first need to install FastFlow and our customized Tensorflow on your GPU/CPU machines following the instructions in this section.
We currently support the Python versions as listed:

#### Supported Python Versions
- 3.7
- 3.8

1) If you installed tensorflow already, remove them to install our customized tensorflow core.

```bash
$ pip uninstall tensorflow
```

2) To execute FastFlow, first you must install our customized TensorFlow core for FastFlow.
https://github.com/SamsungLabs/fastflow-tensorflow

After building the package, copy the .wheel file to your GPU and CPU machines, and install it by executing the command below. 
(Caution: You must build Tensorflow on the same version of platform, such as Ubuntu and Python, with your GPU and CPU machines.)

```bash
$ pip install <custom-tensorflow-wheel-file-path>
```

By importing tensorflow on Python interpreter, you can check whether the installation is done successfully.

```bash
$ python
>>> import tensorflow
```

3) After installation of the customized TensorFlow core, it is time to install FastFlow.
Before installing it, you first need to install the dependencies as shown below:

```bash
$ pip install -r ./requirments.txt
```

Next, to install FastFlow, execute build_pip_package.sh

```bash
$ ./build_pip_package.sh

The shell script build FastFlow package (.whl file) based on the source code in your local repository,
and install it on your machine automatically.
```

## How to Use FastFlow
### 1) FastFlow on Allocated Resources
#### Prerequesites
1) Local GPU machine and remote CPU machines for offloading
2) Dataset on a shared storage or dataset must be copied to the remote CPU nodes


First, launch a tf.data.service dispatcher and workers on the remote CPU nodes.
You need to launch a worker for each remote machine.
Please see [the manual](https://www.tensorflow.org/api_docs/python/tf/data/experimental/service) for tf.data.service: 

```python
import tensorflow as tf

d_config = tf.data.experimental.service.DispatcherConfig(port=5000)
dispatcher = tf.data.experimental.service.DispatchServer(d_config)

w_port = 5001
w_config = tf.data.experimental.service.WorkerConfig(
    dispatcher_address=dispatcher.target.split("://")[1],
    worker_address="<ip-address-of-this-machine>" + ":" + str(w_port),
    port=w_port)
worker = tf.data.experimental.service.WorkerServer(w_config)

dispatcher.join()

```

Please remember the address of the dispatcher because it needs to be provided to the FastFlow config.

After launching a dispatcher and workers in the remote nodes,
you need to convert your training code (written by Keras) using FastFlow's API.
You need to change the following things:
1) import fastflow
2) Use ff.Model instead of keras.Model
3) Define __deepcopy()__ method within your model class that generates a copied model.
4) Add a FastFlow config before model.fit()

Here is the example of how to convert the original Keras code into FastFlow code.
#### Keras code
```python
import tensorflow as tf 
# define input pipeline
ds = tf.data.Dataset(path)
ds = ds.map(..).batch(..).prefetch(..)

# define model 
class MyModel(keras.Model):
    def __init__(self):
        ...

model = MyModel()
model.compile(..)
model.fit(ds, epoch=..)
```

#### FastFlow code
```python
import fastflow as ff
import tensorflow as tf 
# define input pipeline 
ds = tf.data.Dataset(path) # this path must be shared path or the dataset must be in the same path among remote workers 
ds = ds.map(..).batch(..).prefetch(..)

# define model
class MyModel(ff.Model):
    def __init__(self):
        ...

    def __deepcopy__(self):
        ...
        return MyModel()

model = MyModel()
model.compile(..)

# add FastFlow configuration 
config = ff.FastFlowConfig.from_yaml(yaml_path)
model.fit(ds, epoch=..., auto_offload_conf=config)
```

Here is the example of config: 

```yaml
dispatcher_addr: 0.0.0.0 # set the dispatcher address here
dispatcher_port: 0 # dispatcher port 
num_profile_steps: 100 # number of profiling steps
num_initial_steps: 10 # number of initial steps to skip metric profiling
```



### 2) FastFlow on AWS
FastFlow also supports autoscaling on AWS, so you don't have to launch and allocate remote CPU resources before the execution.
For executing FastFlow on AWS, you need to provide some additional configurations. 

#### Prerequesites
1) GPU (EC2) instance where FastFlow and our custom Tensorflow are installed.
2) FastFlow-installed Linux AMI for EC2 instances
3) Dataset in a shared storage such as AWS EFS, mounted to the GPU/CPU instances

#### Build AMI for FastFlow
First, start a CPU instance from a Ubuntu AMI which Python are installed. 
Please refer here to find the Python versions FastFlow supports.
On the instance, install fastflow and fastflow-tensorflow following the process described in [Install](#install) section.

Next, create an AMI from the instance, which will be used by autoscaler in FastFlow, to launch multiple CPU instances for offloading. 
For more detailed instructions, please refer to [AWS EC2 official guide](https://docs.aws.amazon.com/toolkit-for-visual-studio/latest/user-guide/tkv-create-ami-from-instance.html). 

#### Enable Autoscaler in FastFlow
On your GPU instance, you can train your model using FastFlow basically, as explained in [FastFlow Code](#fastflow-code) section.
To use autoscaling for offloading, you should add related options to your configuration as follows.

First, enable autoscaler in FastFlow by adding the following configurations in your config.yaml file.
You can leave `dispatcher_addr` and `dispatcher_port` empty, since they are not used in autoscaling.
```yaml
autoscale_enabled: True
```

You also need to enter another yaml file which contains the configuration for autoscaling on AWS.
For more detailed information, please refer to the doc string of [AWSEC2InstanceOption](fastflow/autoscaler/framework/instance.py).
(Note that FastFlow determines the number of workers launched for offloading,
based on the performance profiling result in your training environment and workload)
```yaml
# An example of instance_option.yaml
image_type: fastflow
boto3_credential: 
    region_name: <your-region-name>
    endpoint_url: <the-endpoint-url-of-your-region>
ec2_instance_option:
    dispatcher:
        ImageId: <your-AMI-id>  # the id of AMI you build
        InstanceType: <ec2-instance-type-for-dispatcher>  # ex. c5.4xlarge
        KeyName: <your-ssh-key-name>
        SecurityGroupIds: 
            - <your-security-group-id>
            - ...
    worker:
        ImageId: <your-AMI-id>
        InstanceType: <ec2-instance-type-for-worker>  # ex. c5.4xlarge
        KeyName: <your-ssh-key-name>
        SecurityGroupIds: 
            - <your-security-group-id>
            - ...
```

Next, read your config yaml file and enter to FastFlow in your code as shown below: 
```python
# add FastFlow configuration 
config = ff.FastFlowConfig.from_yaml(yaml_path)
config.instance_option = ff.instance_option_from_yaml(
    instance_option_yaml_path,
    'aws_ec2')
model.fit(ds, epoch=..., auto_offload_conf=config)
```

Now, during the model training, FastFlow launches and manages ec2 instances for dispatcher and workers automatically.
(Note that, for autoscaling on AWS, you must grant a proper IAM role, `AWSEC2FullAccess` for example, to your GPU instance for starting/terminating EC2 instances on it.) 

---------------


This work is licensed under a [Creative Commons Attribution 4.0 International License][cc-by-nc-nd].

