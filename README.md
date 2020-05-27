## Coronavirus (COVID-19) CC-19 dataset 

We introduce a small new dataset related to the latest family of coronavirus i.e. COVID-19. Such datasets play an important role in the domain of artificial intelligence for clinical medicine related applications. This data set contains the Computed Tomography scan (CT) slices for 89 subjects. Out of these 89 subjects, 68 were confirmed patients (positive cases) of the COVID-19 virus, and the rest 21 were found to be negative cases. The proposed dataset  “CC-19” contains 34,006 CT scan slices (images) belonging to 98 subjects out of which 28,395 CT scan slices belong to positive COVID patients. This dataset is made publically. The first figure shows some 2D slices taken from CT scans of the CC-19 dataset. Moreover, some selected 3D samples from the dataset are shown in Figure. The Hounsfield unit (HU) is the measurement of CT scans radiodensity as shown in Table. Usually, CT scanning devices are carefully calibrated to measure the HU units. This unit can be employed to extract the relevant information in CT Scan slices. The CT scan slices have cylindrical scanning bounds. For unknown reasons, the pixel information that lies outside this cylindrical bound was automatically discarded by the CT scanner system. But fortunately, this discarding of outer pixels eliminates some steps for preprocessing.

##### 2D slice samples
Some random samples of  CT scan 2D slices taken from CC-19 dataset.


![Some random samples of  CT scan 2D slices taken from CC-19 dataset.](https://github.com/abdkhanstd/COVID-19/blob/master/Images/2.png)


##### 3D Volume samples 
This figure shows some selected samples form the “CC-19” dataset.  Each row represents different patient samples with various Hounsfield Unit (HU) for CT scans. The first column, from left to right, shows the lungs in the 3D volumetric CT scan sphere.  The second column shows the extracted bone structure using various HU values followed by the XY, XZ, and YZ plane view of the subjects' CT scan. It is worth noting that the 3D volumetric representation is not pre-processed to remove noise and redundant information.


![This figure shows some selected samples form the “CC-19” dataset.  Each row represents different patient samples with various Hounsfield Unit (HU) for CT scans. The first column, from left to right, shows the lungs in the 3D volumetric CT scan sphere.  The second column shows the extracted bone structure using various HU values followed by the XY, XZ, and YZ plane view of the subjects' CT scan. It is worth noting that the 3D volumetric representation is not pre-processed to remove noise and redundant information.](https://github.com/abdkhanstd/COVID-19/blob/master/Images/1.png)

Collecting datasets is a challenging task as there are many ethical and privacy concerns observed the hospitals and medical practitioners. Keeping in view these norms, this dataset was collected in the earlier days of the epidemic form various hospitals in Chengdu, the capital city of Sichuan. Initially,  the dataset was in an extremely raw form. We preprocessed the data and found many discrepancies with most of the collected CT scans. Finally, the CT scans, with discrepancies, were discarded from the proposed dataset. All the CT scans are different from each other i.e. CT scans have a different number of slices for different patients. We believe that the possible reasons behind the altering number of slices are the difference in height and body structure of the patients. Moreover, upon inspecting various literature, we found that the volume of the lungs of an adult female is, comparatively, ten to twelve percent smaller than a male of the same height and age.


##### Preprocessing guideines
Various values of Hounsfield unit (HU) for different substances.
![Various values of Hounsfield unit (HU) for different substances.](https://github.com/abdkhanstd/COVID-19/blob/master/Images/3.png)

##### Modified Inception V3 and Capsule base Model
Deep Learning Model for COVID-19. We employ a modified version of the inception V3 (IV3*)deep learning model as a feature extraction pipeline. Further,
we train the extracted features using to layers of the capsule network.

![Deep Learning Model for COVID-19. We employ a modified version of the inception V3 (IV3*)deep learning model as a feature extraction pipeline. Further,
we train the extracted features using to layers of the capsule network.](https://github.com/abdkhanstd/COVID-19/blob/master/Images/5.png)
##### Performance of Deep learning models
The performance of some famous deep learning networks. The bold values represent the best performance. It can be seen that the capsule network exhibited the highest sensitivity while ResNet 0.249 has the best specificity.

![The performance of some famous deep learning networks. The bold values represent the best performance. It can be seen that the capsule network exhibited the highest sensitivity while ResNet 0.249 has the best specificity.](https://github.com/abdkhanstd/COVID-19/blob/master/Images/4.png)

#### Download dataset
The data set is about 16GB uncompressed. The compressed version of the dataset is shared via One drive 10GB approximately. The files contain 3D volume or CT scan slices of 89 subjects. The data set can be downloaded using this link.

#### We are bargaining for 30,000+ patients data. Hopefully we will add and upload the data soon.


##### Further information will be updated later on.
##### How to run Python code?
##### Test and Train splits
##### Federated Learning Code
##### Preorecessing 3D volumes
##### How to request dataset?
##### Cite As:
