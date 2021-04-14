from __future__ import print_function
import sys
from random import shuffle

import os
import glob
import numpy as np
import tensorflow as tf
import csv
import pydicom
import cv2
import configparser



def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


default_pixel_spacing = 0.78125

def construct_tfrecord_2d(file_info_list,output_file):
    '''construct a tfrecord file using file info list
        written to construct the dataset containing both ct and mri
        zhe zhu 11/24/2019
    '''
    count = 0
    with tf.python_io.TFRecordWriter(output_file) as writer:
        for file_info in file_info_list:
            count += 1
            dicom_file = file_info['dicom_file']
            vol_label = file_info['label']
            if count % 1000 == 0:
                print('{0} dicom files have been processed. Current: {1} {2}'.format(count, dicom_file, vol_label))
            ds = pydicom.dcmread(dicom_file)
            img = ds.pixel_array
            if img.dtype != np.int16:
                # print('#1 Error! Not Int16 data type {}'.format(dicom_file))
                if img.dtype != np.uint16:
                    print('#1 Error! Data Type:{} type {}'.format(img.dtype, dicom_file))
                img = img.astype(np.int16)
            true_height = img.shape[0]
            true_width = img.shape[1]
            img_height = ds.Rows
            img_width = ds.Columns
            gray_scale = True
            if len(img.shape) != 2:
                gray_scale = False
            if true_height != img_height or true_width != img_width and gray_scale:
                print('#2 Error! {}: from dicom header:{} {} MRI data:{} {}'.format(dicom_file,
                                                                                       img_height, img_width,
                                                                                       true_height, true_width))
            if not gray_scale:
                print('#3 Error! Not 2 channel image: {}'.format(dicom_file))
                img = img[:, :, 0]
            img_raw = img.tostring()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'height': _int64_feature(true_height),
                        'width': _int64_feature(true_width),
                        'label': _int64_feature(vol_label),
                        # 'filename': _bytes_feature(dicom_file),
                        'img_raw': _bytes_feature(img_raw)
                    }
                )
            )
            writer.write(example.SerializeToString())

def build_duke_abdominal_tfrecord():
    '''
    Build the Duke Abdominal Dataset to tfrecord format for training.
    Before running this script, please change !!dataset_folder!! and !!output_folder!!
    '''
    print("Building Duke Abdominal tfrecord...")
    dataset_folder = '/mnt/sdc/dataset/Duke_Abdominal_Deidentified_V3' # Change this to the directory where you put the dataset
    annotation_file = 'labels.txt'
    output_folder = '/mnt/sdc/dataset/tf/2d'  # Change this to the directory on your own computer
    train_info_file = 'train.txt'
    train_tfrecord_file = os.path.join(output_folder,'train.tfrecord')
    val_tfrecord_file = os.path.join(output_folder,'validation.tfrecord')

    # load train pid
    train_info_dict = {}
    with open(train_info_file,'r') as reader:
        lines = reader.readlines()
        for line in lines:
            train_info_dict[line.rstrip()] = 1


    # Load annotation
    labels = {}
    with open(annotation_file,'r') as reader:
        lines = reader.readlines()
        for line in lines:
            pID,examID,seriesID,l = line.split(',')
            if pID not in labels:
                labels[pID] = {}
            if examID not in labels[pID]:
                labels[pID][examID] = {}
            if seriesID not in labels[pID][examID]:
                labels[pID][examID][seriesID] = {}
            labels[pID][examID][seriesID] = int(l)

    # training/validation set
    train_file_info_list = []
    val_file_info_list = []
    patient_folder_list = glob.glob(dataset_folder+"/*")
    for patient_folder in patient_folder_list:
        pID = os.path.basename(patient_folder)
        exam_folder_list = glob.glob(patient_folder+"/*")
        for exam_folder in exam_folder_list:
            examID = os.path.basename(exam_folder)
            series_folder_list = glob.glob(exam_folder+"/*")
            for series_folder in series_folder_list:
                seriesID = os.path.basename(series_folder)
                seriesID = seriesID.replace(" ","")
                dicom_file_list = glob.glob(series_folder+"/*")
                if seriesID not in labels[pID][examID]:
                    print(pID+" "+seriesID)
                label = labels[pID][examID][seriesID]
                for dicom_file in dicom_file_list:
                    file_info = {}
                    file_info['dicom_file'] = dicom_file
                    file_info['label'] = label
                    if pID in train_info_dict:
                        ## Training set
                        train_file_info_list.append(file_info)
                    else:
                        ## Belong to the validation set
                        val_file_info_list.append(file_info)

    # construct train/val tfrecord
    construct_tfrecord_2d(train_file_info_list,train_tfrecord_file)
    construct_tfrecord_2d(val_file_info_list,val_tfrecord_file)

    print("Done")




if __name__=="__main__":
    build_duke_abdominal_tfrecord()