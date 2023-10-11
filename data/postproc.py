# trace generated using paraview version 5.9.1-RC1

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ma", default=0.8395, type=float)
parser.add_argument("--re", default=11.72e6, type=float)
parser.add_argument("--a", default=3.06, type=float)
args = parser.parse_args()
ma = args.ma
re = args.re
a = args.a

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML Unstructured Grid Reader'
filepath = './ma_{:g}/re_{:g}/a_{:g}'.format(ma,re,a)
flowvtu = XMLUnstructuredGridReader(registrationName='flow.vtu', FileName=[filepath+'/flow.vtu'])
flowvtu.PointArrayStatus = ['Density', 'Momentum', 'Energy']

# 5 slices: {root, 0.25b, 0.5b, 0.75b, tip}
# onera m6:
b = 1.1963
# mac_y = 0.539218234272

# sorted
slice_ys = [0.01*b, 0.25*b, 0.5*b, 0.75*b, b]    #paraview selects no nodes if y=0

# create a new 'Slice'
slice1 = Slice(registrationName='Slice1', Input=flowvtu)
slice1.SliceType = 'Plane'
slice1.HyperTreeGridSlicer = 'Plane'
slice1.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slice1.SliceType.Origin = [0.2014751434326172, 5.5, 0.0]

# init the 'Plane' selected for 'HyperTreeGridSlicer'
slice1.HyperTreeGridSlicer.Origin = [0.2014751434326172, 5.5, 0.0]

# Properties modified on slice1
slice1.Triangulatetheslice = 0

# set active source
SetActiveSource(slice1)

def save_slice_stl(i,y):
    # Properties modified on slice1.SliceType
    slice1.SliceType.Origin = [0.0, y, 0.0]
    slice1.SliceType.Normal = [0.0, 1.0, 0.0]

    # save data
    SaveData(filepath+'/slice_{:g}.vtk'.format(i), proxy=slice1, ChooseArraysToWrite=1,
        PointDataArrays=['Density', 'Momentum', 'Energy'])
    
for i,y in enumerate(slice_ys):
    save_slice_stl(i,y)
