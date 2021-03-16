




## Introduction :

Sementic segmentation of Spleen dataset. We have used only single volume of spleen MRI scan.A 2d unet is being used . So data volume has been sliced , to shape of single channel along a choosen axis preferable across coronal axis. 

## Model Network :
- UNet
- its a encoder-decoder network
- input shape of 512 x 512.
- output shape is 512 x 512
- skip connection between encoder and decoder network.


## Dataset:
### Spleen 

- bits per pixel: 32
- Spatial unit: mm, Temporal unit: sec
- Grid is regular with grid spacings: [1.       0.794922 0.794922 5.       0.       0.       0.       0.      ]
- Sagittal: 0, Axial: 1, Coronal: 2
- the volume (in mm³) of a spleen: 96672
- one outlier above 500000
- image_data.shape: (512, 512, 90), label_data.shape: (512, 512, 90) 
- histogram of voxels in the images .


## Results :
| Model | image-size | learning rate | dice | jaccard | 
| ------| -----------| ------------- | -----| --------|
| unet  | 64x64      |0.00001        |0.0   |0.00000  |
| unet  | 64x64      |0.00001        |0.0   |0.00000  |
| unet  | 64x64      |0.00001        |0.0   |0.00000  |
| unet  | 64x64      |0.00001        |0.0   |0.00000  |
| unet  | 64x64      |0.00001        |0.0   |0.00000  |
| unet  | 64x64      |0.00001        |0.0   |0.00000  |

### model is stored in :
- model.py (its a 2d unet model)

### to train:
- python training.py

### to do inference :
- python infernece.py
- models are stored in models folders
- data is put in data folders.
- output volume from the prediction is stored in data folder as out.nii.gz format

### to evaluate the performace :
- python evaluation.py

# 3D MEDICAL IMAGING :

# 3D Medical Imaging - Clinical Fundamentals

## Modelities in 3D imaging :
- CT 
- MRI
- SPECT
- ULTRASOUND
### vocab
- 1. Imaging modality: a device used to acquire a medical image
- 2. MRI scanner: Magnetic Resonance Imaging scanner
- 3. Contrast resolution: the ability of an imaging modality to distinguish between differences in image intensity
- 4. Spatial resolution: the ability of an imaging modality to differentiate between smaller objects

## People (end users) using 3d imaging :
- 1. Primary users : 
- 2. Secialist :
- 3. Surgeons
- 4. Radiologist : they are the imaging specialist

## Some usecases of 3d medical imageing 
- 1. Diagonisis
- 2. Prognosis
- 3. Procedural process such as surgery.

## Diagnostic performance metric:
- Bayes theorem :
P(A|B) = P(A)* P(B|A)/P(B).
It can translated as ,
- Posterior odds = Prior odds * Likehood ratio


- 1.  Likihood ratio :A diagnostic performance metric used to determine whether a test result usefully changes the probability that a condition/disease is present.range is 0 to infinite.1 means not helpful . less than 1 means less likely the disease is present , larger than 1 means , more likely disease is present.


- 2. Accuracy:
- 3. Sensitivity:The probability of a test being positive, given that a patient has the underlying disease/condition.
- 4. Specitivity:The probability of a test being negative, given that a patient does not have the underlying disease/condition.
- 5. Dice:
## Choosing a clinical problem and framing it as a machine learning task.

| Steps                            |               Example              |  
|----------------------------------| -----------------------------------|
| Choosing a clinical case.        | Dementia due to alzhemima disease progression|
| Background research.             | find some scientific papers         | 
| Framing the problem as a machine learning task.|Diagonis as a segmentation task for the hippocampas size | 

## Physical Principles of CT Scanners
#### 1. X-rays:
The main operating agents of a CT scanner are X-rays, which are a form of electromagnetic radiation.It lies between gamma ray and UV light.

#### Physical Principles of CT Scanners

X-rays are a form of ionizing radiation, which means that they carry enough energy to detach electrons from atoms. This presents certain health risks, but the short wavelength of this part of the electromagnetic spectrum allows the radiation to interact with the many structures that compose a human body, thus allowing us to measure the amount of photons that reach detectors and make deductions about the structures that were in the way of photons as they were traveling from the source to the detector, with a high precision.

#### CT SCANNERS
As you have seen, the CT scanner operates by projecting X-rays through the subject’s body.it measures
amount of photons passing through the body attenuated by material between the emitter and detector

X-rays get absorbed or scattered by the anatomy and thus detectors measure the amount of this attenuation that happens along each path that the ray is taking. A collimator shapes the beam and ensures that the X-rays only pass through a narrow slice of the object being imaged. Rotation of a source inside a gantry makes sure that projections happen from different angles so that we can get a good 2D representation of the slice. The moving table ensures that multiple such slices are imaged. A collection of slices makes up a 3-dimensional CT image.

###### Sinogram : 
data comming out of ct scanner are called sinogram . Through a algo called backprojection , a single slice of  image is reconstructed .

###### HU scale :
Hounsfield Scale, named after Sir Godfrey Hounsfield who invented modern CT scanners in the 1970s.
Hounsfield Scale maps tissue types to pixel values of CT scans and is essential to understanding CT scans. 0 is water , ranging from -1000 to 1000.

#### MR SCANNERS:
it stands for magnetic resonance. looks like CT scanners , but works in a differnt way. measure the effect of magnetic resonance (electromagnatic). it is much safer than CT , since , no ionaization radiation is emitted. It produces better soft tissue  contrast resolutions .
          MR scanner leverages a basic physical property of protons (charged elementary particles that make up atoms) to align themselves along the vector of magnetic fields. This effect is particularly pronounced in protons that make up hydrogen atoms. Hydrogen atoms make up water molecules, and water makes up to 50-70% of a human body.

The thing with protons is that they possess a property called spin which could be thought of as spinning around an axis. In a normal environment, the direction of this axis is randomly distributed across different protons. In the presence of a strong magnetic field, though, the proton spins get aligned along the direction of the magnetic field, and start precessing (think of what a spinning top that’s lost some of its momentum is doing):
  When an external radiofrequency pulse is applied, of a frequency proportional to the frequency of precession, the protons respond to this pulse in unison, or resonate, and flip the orientation of their spins. Once this pulse is gone, they return to their original orientation (along the static magnetic field).

The way in which protons return to their original orientation is different and depends on the tissue type that protons are a part of.

Since many protons are returning to their original orientation at once, they generate electrical currents in the coils that are placed nearby. Due to the resonance effect these currents are not insignificant and can be measured - these measurements constitute the data about the tissue being studied which is collected by the MRI scanner.

###### Gradient fields:
are used to vary the static magnetic field, and thus precession frequency, spatially. This allows the MR scanner to isolate a part of the body (i.e. a slice) that is being imaging. Further gradient fields are used to isolate information coming from specific locations within a slice.

##### MRI: K-space, Reconstruction, T1 and T2
- K-Space : The currents measured by RF coils get turned into a digital format, and represented as vectors in “K-space”. The concept of K-space goes back to the wave theory in physics and basically defines a space of vectors that describe characteristics of electromagnetic waves.

- Image-recontruction : In our case, these wave vectors carry information about the characteristics of the matter in the space that has been measured. Essentially, these vectors record the spatial frequency of signal intensity, and thus, through the process that involves an Inverse Fourier Transform and a lot of de-noising and other optimizations, get turned into a familiar 2D image that represents a slice through a human body with different anatomy having different pixel intensity. This process is referred to as image reconstruction in MR physics. Typically, image reconstruction is performed on a computer that is directly embedded into an MR scanner, and the problem of optimizing or scaling image reconstruction alone is a very interesting one.
- Due to greater control over the electromagnetic fields, MR scanners can obtain data directly for a 3D volume in a single “sweep”, without having to go slice-by-slice.

- Pulse Sequences: the combination of gradient fields, RF pulses, and aspects of the signal that is getting measured. Together, these are called a pulse sequence.
          Two very common sequences are called “T1-weighted” and “T2-weighted” sequences (technically these two are looking at different aspects of the same combination of electromagnetic fields). T1 produces greater contrast resolution for fat, and T2 produces greater detail in fluids. Quite often, a contrast medium is used along with a T1 sequence to make certain structures stand out. Thus, the gadolinium agent is often used in neuroradiology to improve the visibility of things like tumors and hemorrhages.

### Common 3D imaging data tasks:
- Windowing, Multi-planar reconstruction (MPR), 3D reconstruction, and registration, are common to many problems that involve viewing or processing 3D medical images.
###### Windowing:
mapping high dynamic range of medical images onto the screen-space gray color scale

###### Multi-Planar Reconstruction:
This is where we construct 2-dimensional images in planes that are not the original acquisition plane. Typically one wants to see images in the cardinal planes - axial, coronal and sagittal, which are orthogonal to each other. Sometimes one is also interested in planes which are not orthogonal to the primary acquisition plane, and in such case we talk about oblique MPR.

###### 3D Reconstruction:
constructing a 3D model from multiple slices of 3D medical imaging data.voxel-based or volumetric reconstruction is quite often used in medical images to reconstruct 3D images.
- One will be a primary plane, where as other two will be reconstructed planes.

###### Registration: 
Some time we need to superimpose anatomy from  differnt modalites.it the process of shifting all the voxels from one image (known as the moving image ) to a another image known as fixed image so that some constraints are  fulfilled . 
- rigid registration  : rotation , translation 
- affine registration : has to do with rescaling
- arbitary registration
- eg CT scan on top of MRI scan. 

## NIFTI :

Like DICOM, NIFTI, which stands for Neuroimaging Informatics Technology Initiative, is an open standard that is available at https://nifti.nimh.nih.gov/nifti-2. The standard has started out as a format to store neurological imaging data and has slowly seen a larger adoption across other types of biomedical imaging fields.

Some things that distinguish NIFTI from DICOM, though are:

NIFTI is optimized to store serial data and thus can store entire image series (and even study) in a single file.
NIFTI is not generated by scanners; therefore, it does not define nearly as many data elements as DICOM does. Compared to DICOM, there are barely any, and mostly they have to do with geometric aspects of the image. Therefore, NIFTI files by themselves can not constitute a valid patient record but could be used to optimize storage, alongside some sort of patient info database.
NIFTI files have fields that define units of measurements and while DICOM files store all dimensions in mm, it’s always a good idea to check what units of measurement are used by NIFTI.
When addressing voxels, DICOM uses a right-handed coordinate system for X, Y and Z axes, while NIFTI uses a left-handed coordinate system. Something to keep in mind, especially when mixing NIFTI and DICOM data.
Further Resources
Background and history of the NIFTI: https://nifti.nimh.nih.gov/background/
The most “official” reference of NIFTI data fields could be found in this C header file, published on the standard page: https://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h or on this, slightly better-organized page: https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields
A great blog post on NIFTI file format: https://brainder.org/2012/09/23/the-nifti-file-format/
New Vocabulary
NIFTI - Neuroimaging Informatics Technology Initiative, is an open standard that is used to store various biomedical data, including 3D images.

## Important Parameters of Medical Images:

##### Orientation parameters
For DICOM two parameters that define the relative position of a 2D in the 3D space would be:

- (0020,0037) Image Orientation Patient - a parameter that stores two vectors (directional cosines to be precise) that define the orientation of the first row and first column of the image.

- (0020,0032) Image Position Patient - a parameter that stores x, y, and z coordinates of the upper left-hand corner of the image.

Both of these are Type 1 (mandatory) parameters for MR and CT IODs, so it is generally safe to rely on them.

For NIFTI, the same purpose is served by srow_*, qoffset_* vectors.

##### Physical spacing parameters
(0028,0030) Pixel Spacing - two values that store the physical distance between centers of pixels across x and y axes.

(0018,0050) Slice Thickness - thickness of a single slice. Note that this one is a Type 2 (required, but can be zero) parameter for CT and MR data. If you find those unavailable, you can deduce slice thickness from IPP parameters. This can happen if your volume has non-uniform slice thickness.

##### Photometric parameters
There are quite a few of those, as DICOM can store both grayscale and color data, so lots of parameters deal with color palettes. CT and MR images usually have monochrome pixel representation (defined by tag (0028,0004) Photometric Interpretation).

##### Most notable ones of this group are:

(0028,0100) Bits Allocated - parameter that defines the number of bits allocated per pixel (since we have CPUs that operate in bytes, this parameter is always a multiple of 8).

(0028,0101) Bits Stored - parameter that defines the number of bits that are actually used - quite often, you could see Bits Allocated set to 16, but Bits Stored set to 12.

##### Image size parameters
Of worthy mention are parameters that define the size of the 3D volume. There are Type 1 parameters that define the width and height of each 2D slice:

(0020,0010) Rows - this is the height of the slice, in voxels

(0020,0011) Columns - width of the slice, in voxels

##### Both of these need to be consistent across all DICOM files that comprise a series.

Note that there isn’t really anything in DICOM metadata that has to tell you how many slices you have in the series. There are tags that can hint at this (like (0054,0081) Number of Slices, or (0020,0013) Instance Number), but none of them are mandatory, Type 1 tags for CT or MR data. The most reliable way to determine the number of slices in the DICOM series is to look at the number of files that you have, and ideally validate that they make up a correct volume by checking for the consistency of IPP values.

## Basic DICOM Volume EDA:

##### Voxel spacing
DICOM voxels do not have to be perfect cubes (as they are in many computer vision problems). There are DICOM Data Elements that will tell you what exactly are the dimensions of voxels. The most important ones are Pixel Spacing and Slice Thickness. However, there are others, and if your project involves measuring things, make sure you get the transformation right by closely inspecting the tags in your dataset and comparing them with the list of elements in the IOD table for the respective modality.

##### Data ranges
We have seen how with CT, you may have data in your dataset that will represent synthetic material or items artificially added by scanners. It is always a good idea to see if there is something outstanding in the image you are dealing with and if it represents something that you need to think about in your downstream processing.

Conversions between DICOM values and screen space are particularly important if you are planning to visualize slices for any kind of diagnostic use or overlay them on top of diagnostic information. We have not really touched the aspects of visualization other than being mindful of bit depth and doing our own windowing, but DICOM images contain quite a lot of information that defines how exactly you are expected to map the data to the screen colorspace. If you are interested in exploring this further or need to accurately represent the data, take a closer look at elements in DICOM’s ImagePixel module. Things like Pixel Representation, Photometric Interpretation, Rescale Slope, Rescale Intercept and many others define how values should be transformed for accurate representation.


# 3D imaging EDA summary :

## Vocabulary
- DICOM - Digital Imaging and Communication in Medicine. The standard defining the storage and communication of medical images.
- DICOM Information Object - representation of a real-world object (such as an MRI scan) per DICOM standard.
- IOD - Information Object Definition. Definition of an information object. Information Object Definition specifies what metadata fields have to be in place for a DICOM Information Object to be valid. IODs are published in the DICOM standard.
- Patient - a subject undergoing the imaging study.
- Study - a representation of a “medical study” performed on a patient. You can think of a study as a single visit to a hospital for the purpose of taking one or more images, usually within. A - Study contains one or more series.
- Series - a representation of a single “acquisition sweep”. I.e., a CT scanner took multiple slices to compose a 3D image would be one image series. A set of MRI T1 images at different axial levels would also be called one image series. Series, among other things, consists of one or more instances.
- Instance - (or Image Information Entity instance) is an entity that represents a single scan, like a 2D image that is a result of filtered backprojection from CT or reconstruction at a given level for MR. Instances contain pixel data and metadata (Data Elements in DICOM lingo).
- SOP - Service-Object Pair. DICOM standard defines the concept of an Information Object, which is the representation of a real-world persistent object, such as an MRI image (DICOM Information Objects consist of Information Entities).
- Data Element - a DICOM metadata “field”, which is uniquely identified by a tuple of integer numbers called group id and element id.
- VR - Value Representation. This is the data type of a DICOM data element.
- Data Element Type - identifiers that are used by Information Object Definitions to specify if Data Elements are mandatory, conditional or optional.
- NIFTI - Neuroimaging Informatics Technology Initiative, is an open standard that is used to store various biomedical data, including 3D images.



# 3D Medical Imaging - End-to-End Deep Learning Applications:

## Classification: Introduction and Use Cases
- problems that lend themselves well to solutions via automated classification or object detection algorithm.

- Detecting brain hemorrhages, or bleedings in the brain is particularly important in emergency scenarios when brain damage can happen within minutes. Often, radiologists have a backlog of images that they are going through, and it is not obvious which ones should be prioritized. An algorithm that will spot time-critical conditions will help with such prioritization

- Screening and monitoring scenarios, such as the presented scenario of screening for lung nodules, can be quite tedious because objects that are sought can hide well, and meticulous scrolling through slices is required. Pointing human attention to areas which are likely to be suspicious is helpful and saves time

- The presented scenario of incidental findings deals with an interesting phenomenon of selective attention where humans tend to ignore certain stimuli when multiple are applied. Thus, even trained observers may ignore something otherwise quite obvious, like an adrenal cyst when they know that image was taken with the purpose of evaluating potential vertebral disc degeneration. The famous “gorilla study” represents this marvelously.

Note: when choosing a medical imaging problem to be solved by machine learning, it is tempting to assume that automated detection of certain conditions would be the most valuable thing to solve. However, this is not usually the case. Quite often detecting if a condition is present is not so difficult for a human observer who is already looking for such a condition. Things that bring most value usually lie in the area of productivity increase. Helping prioritize the more important exams, helping focus the attention of a human reader on small things or speed up tedious tasks usually is much more valuable. Therefore it is important to understand the clinical use case that the algorithm will be used well and think of end-user value first.

- When it comes to classification and object detection problems, the key to solving those is identifying relevant features in the images, or feature extraction. Not so long ago, machine learning methods relied on manual feature design. With the advent of CNNs, feature extraction is done automatically by the network, and the job of a machine learning engineer is to define the general shape of such features. As the name implies, features in Convolutional Neural Networks take the shape of convolutions. In the next section, let’s take a closer look at some of the types of convolutions that are used for 3D medical image analysis.

###### New Vocabulary
1. Classification - the problem of determining which one of several classes an image belongs to.
2. Object Detection - the problem of finding a (typically rectangular) region within an image that matches with one of several classes of interest.

## Methods for Feature Extraction
- 2D Convolution is an operation visualized in the image above, where a convolutional filter is applied to a single 2D image. Applying a 2D convolution approach to a 3D medical image would mean applying it to every single slice of the image. A neural network can be constructed to either process slices one at a time, or to stack such convolutions into a stack of 2D feature maps. Such an approach is fastest of all and uses least memory, but fails to use any information about the topology of the image in the 3rd dimension.

- 2.5D Convolution is an approach where 2D convolutions are applied independently to areas around each voxel (either in neighboring planes or in orthogonal planes) and their results are summed up to form a 2D feature map. Such an approach leverages some 3-dimensional information.

- 3D Convolution is an approach where the convolutional kernel is 3 dimensional and thus combines information from all 3 dimensions into the feature map. This approach leverages the 3-dimensional nature of the image, but uses the most memory and compute resources.

- Understanding these is essential to being able to put together efficient deep neural networks where convolutions together with downsampling are used to extract higher-order semantic features from the image.

## Segmentation: Introduction and Use Cases

a few use cases for segmentation in 3D medical imaging:

- Longitudinal follow up: Measuring volumes of things and monitoring how they change over time. These methods are very valuable in, e.g., oncology for tracking slow-growing tumors.

- Quantifying disease severity: Quite often, it is possible to identify structures in the organism whose size correlates well with the progression of the disease. For example, the size of the hippocampus can tell clinicians about the progression of Alzheimer's disease.

- Radiation Therapy Planning: One of the methods of treating cancer is exposing the tumor to ionizing radiation. In order to target the radiation, an accurate plan has to be created first, and this plan requires careful delineation of all affected organs on a CT scan

- Novel Scenarios: Segmentation is a tedious process that is not quite often done in clinical practice. However, knowing the sizes and extents of the objects holds a lot of promise, especially when combined with other data types. Thus, the field of radiogenomics refers to the study of how the quantitative information obtained from radiological images can be combined with the genetic-molecular features of the organism to discover information not possible before.

- Segmentation - the problem of identifying which specific pixels within an image belong to a certain object of interest.

##  Segmentation Methods
-A U-Net architecture has been very successful in analyzing 3D medical images and has spawned multiple offshoots. 

### Performance metrices for segmentation task are 

- Sensitivity and Specificity
- Dice Similarity Coefficient
- Jaccard Index
- Hausdorff Distance


# Tools and libraries

We tried to minimize the dependency on external libraries and focus on understanding some key concepts. At the same time, there are many tools that the community has developed, which will help you get moving faster with the tasks typical for medical imaging ML workflows.

A few tools/repos worthy of attention are:

Fast.ai - python library for medical image analysis, with focus on ML: https://dev.fast.ai/medical.imaging
MedPy - a library for medical image processing with lots of various higher-order processing methods: https://pypi.org/project/MedPy/
Deepmedic, a library for 3D CNNs for medical image segmentation: https://github.com/deepmedic/deepmedic
Work by the German Cancer Research Institute:
https://github.com/MIC-DKFZ/trixi - a boilerplate for machine learning experiment
https://github.com/MIC-DKFZ/batchgenerators - tooling for data augmentation
A publication about a project dedicated to large-scale medical imaging ML model evaluation which includes a comprehensive overview of annotation tools and related problems (including inter-observer variability): https://link.springer.com/chapter/10.1007%2F978-3-319-49644-3_4


# Deploying AI Algorithms in Real World Scenarios:

## DICOM Networking: Introduction:
When it comes to moving medical images around the hospital, the DICOM standard comes to the rescue. Alongside the definition of the format for storing images and metadata (which we have looked at in detail in previous lessons, it defines the networking protocol for moving the images around.

## DICOM Networking: Services
1. There are two types of DICOM networking: DIMSE (DICOM Message Service Element) and DICOMWeb. The former is designed to support data exchange in protected clinical networks that are largely isolated from the Internet. The latter is a set of RESTful APIs (link to the Standard) that are designed to communicate over the Internet. DIMSE networking does not have a notion of authentication and is prevalent inside hospitals.

2. DIMSE networking defines how DICOM Application Entities talk to each other on protected networks. 
3. DICOM Application Entities that talk to each other take on roles of Service Class Providers which are an an AE that provides services over DIMSE network and
* Service Class Users which is an AI that requests service from an SCP
4. SCPs typical respond to requests and SCUs issue them
5. Full list of DIMSE services could be found in the Part 7 of the DICOM Standard, ones that you are most likely run into are:
6. C-Echo - “DICOM ping” - checks if the other party can speak DICOM
7. C-Store - request to store an instance
8. An Application Entity (AE) is an actor on a network (e.g. a medical imaging modality or a PACS) that can talk DIMSE messages defined by three parameters:
- Port
- IP Address
- Application Entity Title (AET) - an alphanumeric string



# Referneces :

1. Udacity github repositories
2. AI in Healthcare course (Udacity)
