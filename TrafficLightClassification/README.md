https://github.com/bosch-ros-pkg/bstld

### Training the model
#### Install Linux
```
sudo apt-get update
pip install --upgrade dask
```

For Cuda 9
```
pip install tensorflow-gpu==1.12
```
For Cuda 8
```
pip install tensorflow-gpu==1.4 
```

Additional packages
```
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
```

Create directory for training code, model and data
```
mkdir TrafficLightClassification
cd TrafficLightClassification
```

Get models from tensorflows model repository that are compatible with tensorflow 1.4
```
git clone https://github.com/tensorflow/models.git
cd models
git checkout f7e99c0
```

Test the installation
```
cd research
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
python object_detection/builders/model_builder_test.py
```
#### Data
#####Real World
Images with labeled traffic lights can be found on

1.  [Bosch Small Traffic Lights Dataset](https://hci.iwr.uni-heidelberg.de/node/6132)
2.  [LaRA Traffic Lights Recognition Dataset](http://www.lara.prd.fr/benchmarks/trafficlightsrecognition)
3.  Udacity's ROSbag file from Carla
4.  Traffic lights from Udacity's simulator

##### Simulation
Training images for simulation can be found downloaded from Vatsal Srivastava's dataset and Alex Lechners's dataset. The images are already labeled and a  [TFRecord file](https://github.com/alex-lechner/Traffic-Light-Classification#23-create-a-tfrecord-file)  is provided as well:

1.  [Vatsal's dataset](https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI#get-the-dataset)
2.  [Alex Lechner's dataset](https://www.dropbox.com/s/vaniv8eqna89r20/alex-lechner-udacity-traffic-light-dataset.zip?dl=0)

#### Model 

The model "SSD Mobilenet V1" was used for classification of the Bosch Small Traffic Lights Dataset. See the performance on this page https://github.com/bosch-ros-pkg/bstld .

The model "SSD Inception V2" seems to perform better at the expense of speed. See [Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

#### Download
switch to the models directory and download 
```
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz

wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
```
extract them there
```
tar -xzf ssd_mobilenet_v1_coco_2018_01_28.tar.gz
tar -xzf ssd_inception_v2_coco_2018_01_28.tar.gz
```

#### Model Configuration

Go back to the TrafficLightClassification directory and create a config directory.

```
mkdir config
```

copy the chosen models to config
```
cp models/research/object_detection/samples/configs/ssd_mobilenet_v1_coco.config config/
cp models/research/object_detection/samples/configs/ssd_inception_v2_coco.config config/
```

##### Cofiguration on Udacity Carla dataset for "SSD Inception V2"

Configuration taken from https://github.com/bosch-ros-pkg/bstld/blob/master/tf_object_detection/configs/ssd_mobilenet_v1.config

1.  Change  `num_classes: 90`  to the number of labels in your  `label_map.pbtxt`. This will be  `num_classes: 4`
2.  Set the default  `max_detections_per_class: 100`  and  `max_total_detections: 300`  values to a lower value for example  `max_detections_per_class: 25`  and  `max_total_detections: 100`
3.  Change  `fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt"`  to the directory where your downloaded model is stored e.g.:  `fine_tune_checkpoint: "models/ssd_inception_v2_coco_2018_01_28/model.ckpt"`
4.  Set  `num_steps: 200000`  down to  `num_steps: 20000`
5.  Change the  `PATH_TO_BE_CONFIGURED`  placeholders in  `input_path`  and  `label_map_path`  to your .record file(s) and  `label_map.pbtxt`

### Train
Copy `train.py` from `TrafficLightClassification/models/research/object_detection` to `TrafficLightClassification` folder

Start Training with 
```
python train.py --logtostderr --train_dir=./models/train --pipeline_config_path=./config/ssd_inception_v2_coco-simulator.config

