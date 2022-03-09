# Human Activity Recognition

The data consists of a number of accelerometer measurements from cellphones. The 9 features are,
- Triaxial acceleration from the accelerometer (total acceleration),
- Estimated body acceleration.\n",
- Triaxial Angular velocity from the gyroscope.
 
 These features are measured when the test subject is performing one of these activites:,
 - Walking,
 - Walking upstairs,
 - Walking downstairs,
 - Sitting,
 - Standing,
 - Laying.

The goal is to correctly classify these activities from time series.
In the jupyter notebook, a full analysis is performed.
A naive baseline achieves 39% accuracy. This is improved to roughtly 80% by using PCA and KNN. A simple random forrest achieves 85% and a deep learning model (CNN-LSTM) acheives 91%. 

Clone repo to your desired folder.
Use `./build.sh` to build a docker image.
Use `./run.sh` to run the docker container with a volume mount. 

