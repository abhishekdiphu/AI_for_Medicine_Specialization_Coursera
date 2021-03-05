




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




# Referneces :

1. Udacity github repositories
2. AI in Healthcare course (Udacity)
