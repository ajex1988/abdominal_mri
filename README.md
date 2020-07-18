# Duke Abdominal MRI Series Dataset
This dataset contains abdominal MRI series for 190 patients, including 134 males and 56 females. There are 190 exams (3717 series) in total. Each series was annotated by two radiologists in consensus with regard to series type. We have annotated 30 series types (details below). This dataset was collected at Duke University Medical Center. 

## Series Types
We provide all the 30 series types in the following table. In each cell the format is *series type*(*numeric label*).
|  |  |   |   |  |
|--------------------------------------------|---------------------------------|-------------------------|--------------------|----------------|
| Coronal Precontrast Fat Suppressed T1w(24) | Coronal Late Dynamic T1w(6)     | Venous Subtraction(29)  | Anythingelse(1)    | Axial T2w(9)   |
| Coronal Transitional/ Hepatocyte T1w(13)   | Axial Late Dynamic T1w(16)      | Portal Venous T1w(23)   | Arterial T1w(2)    | Fat Only(11)   |
| Coronal Steady State Free Precession(28)   | Proton Density Fat Fraction(21) | Coronal In Phase(15)    | Axial In Phase(14) | Axial ADC(0)   |
| Axial Precontrast Fat Suppressed T1w(25)   | Water Density Fat Fraction(22)  | Early Arterial T1w(3)   | Coronal T2w(7)     | Localizers(17) |
| Axial Transitional/ Hepatocyte T1w(12)     | Axial Opposed Phase(19)         | Late Arterial T1w(4)    | Axial DWI(8)       | MRCP(18)       |
| Axial Steady State Free Precession(27)     | Coronal Opposed Phase(20)       | Arterial Subtraction(5) | Coronal DWI(10)    | R2*(26)        |
## Dicom
We provide the dicom file for each slice of the series. In Python, **pydicom** can be used to extract useful labels and the MRI data:
```
import pydicom as dcm

dcm_file = 'path/to/dcm_file'
ds = dcm.dcmread(dcm_file)

slice = ds.pixel_array
...

```
## Annotation
We provide several different formats of the annotation. In *demo.py* we demonstrate how to load the annotation of differents formats.

## Download
The dataset can be download from **here**(coming soon).

## Citation
If you find this dataset is useful, please cite our paper:
```
3D Pyramid Pooling Network for Liver MRI Series Classification.
Zhe Zhu, Amber Mittendorf, Erin Shropshire, Brian Allen, Chad Miller, Mustafa R. Bashir, and Maciej A. Mazurowski, 
IEEE Transactions on Pattern Analysis and Machine Intelligence.
```


## Team
