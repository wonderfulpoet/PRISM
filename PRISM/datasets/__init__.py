import os.path as path
from .deep_lesion import DeepLesionTrain,DeepLesionTest
from .spineweb import SpinewebTrain,SpinewebTest
from .nature_image import NatureImage

def get_dataset_train(dataset_type, **dataset_opts):
    return {
        "deep_lesion": DeepLesionTrain,
        "spineweb": SpinewebTrain,
        "nature_image": NatureImage
    }[dataset_type](**dataset_opts[dataset_type])

def get_dataset_test(dataset_type, **dataset_opts):
    return {
        "deep_lesion": DeepLesionTest,
        "spineweb": SpinewebTest,
        "nature_image": NatureImage
    }[dataset_type](**dataset_opts[dataset_type])
