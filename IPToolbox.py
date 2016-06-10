# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 00:52:46 2016

@author: Karthik
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 09:49:40 2016

@author: Karthik
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 08 12:58:44 2016

@author: Karthik
"""
import Tkinter as TT
import random
import cv2
import math
import numpy

def IterativeThresholding(image_name):
    grayimage = []
    grayimage2 = []
    image = cv2.imread(image_name)
    grayimage = cv2.imread(image_name,-1)
    grayimage2 = cv2.imread(image_name,-1)
    if type(image[0][0]) is list:
        grayimage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        grayimage2 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    def iterator(rows,columns,threshold):
        white_pix = []
        black_pix = []
        for row in range(rows):
            for column in range(columns):
                value = grayimage[row][column]
                if value >= threshold:
                    white_pix += [value]
                else:
                    black_pix += [value]
        black_pixels = len(black_pix)
        white_pixels = len(white_pix)
        average_white = sum(white_pix)/white_pixels
        average_black = sum(black_pix)/black_pixels
        new_threshold = (average_white+average_black)/float(2)
        return new_threshold
    def threshold_image(rows,columns,final_threshold):
        for row in range(rows):
            for column in range(columns):
                if grayimage2[row][column] > final_threshold:
                    grayimage2[row][column] = 255
                else:
                    grayimage2[row][column] = 0
#    image_name = raw_input("Enter the name of the image")
#    grayimage = []
#    grayimage2 = []
#    image = cv2.imread(image_name)
#    grayimage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#    grayimage2 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    rows = len(grayimage)
    columns = len(grayimage[0])
    pix_val = []
    for row in range(rows):
        for column in range(columns):
            pix_val += [grayimage[row][column]]

    maximum = max(pix_val)
    minimum = min(pix_val)

    print minimum,maximum

    threshold_init = (maximum+minimum)/float(2)
    differ = 100

    while differ > 0.2:
        new_threshold = iterator(rows,columns,threshold_init)
        #print new_threshold,threshold_init
        differ = abs(new_threshold - threshold_init)
        threshold_init = new_threshold
    image_name = image_name.split('.')
    disk_name = image_name[0] + "iter.jpg"
    threshold_image(rows,columns,new_threshold)
    cv2.imshow("Prior Iterative Threshold",grayimage)
    cv2.imshow("Post Iterative Threshold",grayimage2)
    cv2.imwrite(disk_name,grayimage2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def enhance(imagetoenhance):
    image = cv2.imread(imagetoenhance)
    grayimage = cv2.imread(imagetoenhance,-1)
    newgrayimage = cv2.imread(imagetoenhance,-1)
    if type(imagetoenhance[0][0]) is list:
        grayimage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        newgrayimage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    base_low = 256
    base_high = 0

    rows = len(grayimage)
    columns = len(grayimage[1])

    for row in range(rows):
        for column in range(columns):
            if grayimage[row][column] > base_high:
                base_high = grayimage[row][column]

    for row in range(rows):
        for column in range(columns):
            if grayimage[row][column] < base_low:
                base_low = grayimage[row][column]

    print "Highest Intensity",base_high,"Lowest Intensity",base_low
    rows = len(grayimage)
    columns = len(grayimage[1])
    difference = base_high - base_low
    for row in range(rows):
        for column in range(columns):
            if newgrayimage[row][column] <= base_low:
                newgrayimage[row][column] = 0
            elif newgrayimage[row][column] >= base_high:
                newgrayimage[row][column] = 255
            else:
                base_diff = (newgrayimage[row][column] - base_low)*255
                newgrayimage[row][column] = base_diff/difference
    image_name = imagetoenhance.split('.')
    disk_name = image_name[0] + "enhance.jpg"
    cv2.imshow("Low Contrast Image",grayimage)
    cv2.imshow("Enhanced Image",newgrayimage)
    cv2.imwrite(disk_name,newgrayimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def medianfilter(image_name,mask_size,white,black):

    def noise_internal(image_name,white,black):
        newimage = cv2.imread(str(image_name))
        newgray = cv2.imread(image_name,-1)
        dupnewgray = cv2.imread(image_name,-1)
        if type(newimage[0][0]) is list:
            newgray = cv2.cvtColor(newimage,cv2.COLOR_BGR2GRAY)
            dupnewgray = cv2.cvtColor(newimage,cv2.COLOR_BGR2GRAY)
        rows = len(newgray)
        columns = len(newgray[1])
        total_pixels = rows*columns
        white = (total_pixels * white)/100
        black = (total_pixels * black)/100
        for i in range(white):
            row = int(random.random()*rows)
            column = int(random.random()*columns)
            newgray[row][column] = 255
        for j in range(black):
            row = int(random.random()*rows)
            column = int(random.random()*columns)
            newgray[row][column] = 0
        return newgray

    image = cv2.imread(image_name)
    grayimage = cv2.imread(image_name,-1)
    grayimage2 = cv2.imread(image_name,-1)
    if type(image[0][0]) is list:
        grayimage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        grayimage2 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    noiseimage = noise_internal(image_name,white,black)
    newnoiseimage = noise_internal(image_name,white,black)

    def getmedianvalue(i,j,mean_value):
        startrow = i - mean_value
        endrow = i + mean_value
        startcolumn = j - mean_value
        endcolumn = j + mean_value
        values_list = []
        for row in range(startrow,endrow+1,1):
            for column in range(startcolumn,endcolumn+1,1):
                values_list += [newnoiseimage[row][column]]
        values_list.sort()
        value = values_list[len(values_list)/2]
        noiseimage[i][j] = value

    M = int(mask_size)
    rows = len(grayimage)-1
    columns = len(grayimage[1])-1
    upperbound = M-1
    lowerbound = 0
    mean = (upperbound + lowerbound)/2
    firstrow = mean
    firstcolumn = mean
    lastrow = rows-mean
    lastcolumn = columns-mean
    for i in range(firstrow,lastrow+1,1):
        for j in range(firstcolumn,lastcolumn+1,1):
            getmedianvalue(i,j,mean)
#    median =cv2.medianBlur(noiseimage,M)
    image_name = image_name.split('.')
    disk_name = image_name[0] + "median.jpg"
    cv2.imshow("Original Image",grayimage)
    cv2.imshow("Image after noise addition",newnoiseimage)
    cv2.imshow("Image after median filtering",noiseimage)
    cv2.imwrite(disk_name,noiseimage)
    #cv2.imshow("NOISE REDUCED MAHESH",median)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#def medianfilter(image_name,kernel_size):
#    def make_noise(image_name):
#        newimage = cv2.imread(image_name)
#        newgray = cv2.cvtColor(newimage,cv2.COLOR_BGR2GRAY)
#        rows = len(newgray)
#        columns = len(newgray[1])
#        noisematrix = [[random.randint(0,255) for i in range(columns)] for j in range(rows)]
#        for i in range(rows):
#            for j in range(columns):
#                if noisematrix[i][j] == 0:
#                    newgray[i][j] = 0
#                elif noisematrix[i][j] == 255:
#                    newgray[i][j] = 255
#        return newgray
#    image = cv2.imread(image_name)
#    grayimage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#    noiseimage = make_noise(image_name)
#    degree = int(kernel_size)
#    median =cv2.medianBlur(noiseimage,degree)
#    cv2.imshow("Gray Image",grayimage)
#    cv2.imshow("Noisy Image",noiseimage)
#    cv2.imshow("Noise Reduced Image",median)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

def averaging(image_name,kernel_size,white,black):
    def make_noise(image_name,white,black):
        newimage = cv2.imread(str(image_name))
        newgray = cv2.imread(image_name,-1)
        dupnewgray = cv2.imread(image_name,-1)
        if type(newimage[0][0]) is list:
            newgray = cv2.cvtColor(newimage,cv2.COLOR_BGR2GRAY)
            dupnewgray = cv2.cvtColor(newimage,cv2.COLOR_BGR2GRAY)
        rows = len(newgray)
        columns = len(newgray[1])
        total_pixels = rows*columns
        white = (total_pixels * white)/100
        black = (total_pixels * black)/100
        for i in range(white):
            row = int(random.random()*rows)
            column = int(random.random()*columns)
            newgray[row][column] = 255
        for j in range(black):
            row = int(random.random()*rows)
            column = int(random.random()*columns)
            newgray[row][column] = 0
        return newgray

    #return newgray
#    def make_noise(image_name):
#        newimage = cv2.imread(image_name)
#        newgray = cv2.cvtColor(newimage,cv2.COLOR_BGR2GRAY)
#        rows = len(newgray)
#        columns = len(newgray[1])
#        noisematrix = [[random.randint(0,255) for i in range(columns)] for j in range(rows)]
#        for i in range(rows):
#            for j in range(columns):
#                if noisematrix[i][j] == 0:
#                    newgray[i][j] = 0
#                elif noisematrix[i][j] == 255:
#                    newgray[i][j] = 255
#        return newgray
    def compute(M,i,j,mean):
        total = 0
        for p in range(i-mean,i+mean+1,1):
            for q in range(j-mean,j+mean+1,1):
                total += noiseimage[p][q]
        return total
    image = cv2.imread(image_name)
    imagegray = cv2.imread(image_name,-1)
    blurimgray = cv2.imread(image_name,-1)
 #   blurim = cv2.imread(image_name)
    if type(image[0][0]) is list:
        imagegray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        blurimgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    noiseimage = make_noise(image_name,white,black)
    newnoiseimage = make_noise(image_name,white,black)
    rows = len(image)-1
    columns = len(image[0])-1
    M = int(kernel_size)
#blurim = cv2.blur(image,(M,M))
    upperbound = M-1
    lowerbound = 0
    mean = (upperbound + lowerbound)/2
    firstrow = mean
    firstcolumn = mean
    lastrow = rows-mean
    lastcolumn = columns-mean
    for i in range(firstrow,lastrow+1,1):
        for j in range(firstcolumn,lastcolumn+1,1):
            total_sum = 0
            total_sum += compute(M,i,j,mean)
            avg = total_sum/(M*M)
            noiseimage[i][j] = avg
    image_name = image_name.split('.')
    disk_name = image_name[0] + "avg.jpg"
    cv2.imshow("Original Image",imagegray)
    cv2.imshow("Image after noise addition",noiseimage)
    cv2.imshow("Image after average filtering",newnoiseimage)
    cv2.imwrite(disk_name,newnoiseimage)

    print rows,columns
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def toBinary(image_name,threshold):
    image = cv2.imread(image_name)
    grayimage = cv2.imread(image_name,-1)
    dupgrayimage = cv2.imread(image_name,-1)
    if type(image[0][0]) is list:
        grayimage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        dupgrayimage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    threshold  = int(threshold)
    rows = len(image)
    columns = len(image[1])
    for i in range(rows):
        for j in range(columns):
            if grayimage[i][j] > threshold:
                grayimage[i][j] = 255
            else:
                grayimage[i][j] = 0

    #cv2.imshow("Original Image",image)
    image_name = image_name.split('.')
    disk_name = image_name[0] + "binary.jpg"
    cv2.imshow("Image after binary thresholding",grayimage)
    cv2.imshow("Gray Image",dupgrayimage)
    cv2.imwrite(disk_name,grayimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def PTile(image_name,Pvalue):
    image = cv2.imread(image_name)
    grayimage = cv2.imread(image_name)
    dupgrayimage = cv2.imread(image_name)
    if type(image[0][0]) is list:
        grayimage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        dupgrayimage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    rows = len(grayimage)
    columns = len(grayimage[1])
    AllPixels = []
    for i in range(rows):
        for j in range(columns):
            AllPixels += [grayimage[i][j]]

    AllPixels.sort()

    P = int(Pvalue)
    pixels = rows*columns
    count = int(P*(0.01)*pixels)

    start = pixels-count-1
    base = AllPixels[start]
    for i in range(rows):
        for j in range(columns):
            if grayimage[i][j] > base:
                grayimage[i][j] = 255
            else:
                grayimage[i][j] = 0
    image_name = image_name.split('.')
    disk_name = image_name[0] + "PTile.jpg"
    cv2.imshow('Original Image',dupgrayimage)
    cv2.imshow('Image after PTile Thresholding',grayimage)
    cv2.imwrite(disk_name,grayimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def noise(image_name,white,black):
    newimage = cv2.imread(str(image_name))
    newgray = cv2.imread(image_name,-1)
    dupnewgray = cv2.imread(image_name,-1)
    if type(newimage[0][0]) is list:
        newgray = cv2.cvtColor(newimage,cv2.COLOR_BGR2GRAY)
        dupnewgray = cv2.cvtColor(newimage,cv2.COLOR_BGR2GRAY)
    rows = len(newgray)
    columns = len(newgray[1])
    total_pixels = rows*columns
    white = (total_pixels * white)/100
    black = (total_pixels * black)/100
    for i in range(white):
        row = int(random.random()*rows)
        column = int(random.random()*columns)
        newgray[row][column] = 255
    for j in range(black):
        row = int(random.random()*rows)
        column = int(random.random()*columns)
        newgray[row][column] = 0

    cv2.imshow("Noise Image",newgray)
    cv2.imshow("Original Image",dupnewgray)
    image_name = image_name.split('.')
    disk_name = image_name[0] + "noise.jpg"
    cv2.imwrite(disk_name,newgray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #return newgray

def Roberts(image_name):
#    image_name = raw_input("ENter image name")
    image = cv2.imread(image_name)
    image2 = cv2.imread(image_name,-1)
    image3 = cv2.imread(image_name,-1)
    print type(image2[0][0])
    if type(image[0][0]) is numpy.ndarray:
        image2 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image3 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    rows = len(image)
    columns = len(image[1])


    for row in range(1,rows-1,1):
        for column in range(1,columns-1,1):
            value1 = image2[row][column]
            value2 = image2[row+1][column+1]
            value3 = image2[row][column+1]
            value4 = image2[row+1][column]
            if value1 >= value2:
                valuex = value1 - value2
            elif value1 < value2:
                valuex =  value2 - value1
            if value3 >= value4:
                valuey = value3 - value4
            elif value3 < value4:
                valuey = value4 - value3

            #print "PIXVALS1",image2[row][column],image2[row+1][column+1]
            #print "PIXVALS",image2[row][column+1],image2[row+1][column]
            #print "XGRAD",valuex,"YGRAD",valuey
            value = int(math.sqrt(math.pow(valuex,2) + math.pow(valuey,2)))
            #print value
            if value>255:
                value = 255
            image3[row][column] = value
    
    image_name = image_name.split('.')
    disk_name = image_name[0] + "robert.jpg"
    cv2.imshow("Original",image2)
    cv2.imshow("Modified",image3)
    cv2.imwrite(disk_name,image3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def Kirsch(image_name):
    sobel_mask_y = [[1,0,-1],[2,0,-2],[1,0,1]]
    sobel_mask_x = [[1,2,1],[0,0,0],[-1,-2,-1]]
#    image_name = raw_input("Enter image name")
    image = cv2.imread(image_name,-1)
    image2 = cv2.imread(image_name,-1)
    image3 = cv2.imread(image_name,-1)
    if type(image[0][0]) is numpy.ndarray:
        image2 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image3 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    rows = len(image)
    columns = len(image[1])

    def kirsch_x(xindex,yindex):
        total = int(image2[xindex-1][yindex]) + int(image2[xindex-1][yindex+1]) + int(image2[xindex][yindex+1])-int(image2[xindex][yindex-1]) - int(image2[xindex+1][yindex-1]) - int(image2[xindex+1][yindex])
        return total

    def kirsch_y(xindex,yindex):
        total = int(image2[xindex-1][yindex]) + int(image2[xindex-1][yindex-1]) + int(image2[xindex][yindex-1])-int(image2[xindex][yindex+1]) - int(image2[xindex+1][yindex+1]) - int(image2[xindex+1][yindex])
        return total

    def prewitt_x(xindex,yindex):
        total = int(image2[xindex-1][yindex+1]) + int(image2[xindex][yindex+1]) +int(image2[xindex+1][yindex+1]) - int(image2[xindex][yindex-1]) - int(image2[xindex-1][yindex-1]) - int(image2[xindex+1][yindex-1])
        return total

    def prewitt_y(xindex,yindex):
        total = int(image2[xindex-1][yindex+1]) + int(image2[xindex-1][yindex]) +int(image2[xindex-1][yindex-1]) - int(image2[xindex+1][yindex]) - int(image2[xindex+1][yindex-1]) - int(image2[xindex+1][yindex+1])
        return total

    for row in range(1,rows-1,1):
        for column in range(1,columns-1,1):
            k_x = kirsch_x(row,column)
            k_y = kirsch_y(row,column)
            p_x = prewitt_x(row,column)
            p_y = prewitt_y(row,column)
            value = int(max(k_x,k_y,p_x,p_y))
            if value>255:
                value = 255
            elif value < 0:
                value = 0
            image3[row][column] = value
    image_name = image_name.split('.')
    disk_name = image_name[0] + "kirsch.jpg"
    cv2.imshow("Original Image",image2)
    cv2.imshow("Kirsch Image",image3)
    cv2.imwrite(disk_name,image3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Sobel(image_name):
    sobel_mask_y = [[1,0,-1],[2,0,-2],[1,0,1]]
    sobel_mask_x = [[1,2,1],[0,0,0],[-1,-2,-1]]
#    image_name = raw_input("Enter image name")
    image = cv2.imread(image_name,-1)
    image2 = cv2.imread(image_name,-1)
    image3 = cv2.imread(image_name,-1)
    if type(image[0][0]) is numpy.ndarray:
        image2 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image3 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    rows = len(image)
    columns = len(image[1])

    def sobel_sum_x(xindex,yindex):
        total = image2[xindex-1][yindex-1] + 2*image2[xindex-1][yindex] + image2[xindex-1][yindex+1]-image2[xindex+1][yindex-1] - 2*image2[xindex+1][yindex] - image2[xindex+1][yindex+1]
        return total

    def sobel_sum_y(xindex,yindex):
        total = image2[xindex-1][yindex-1]+2*image2[xindex][yindex-1]+image2[xindex+1][yindex-1]-image2[xindex+1][yindex+1]-2*image2[xindex][yindex+1]-image2[xindex+1][yindex+1]
        return total

    for row in range(1,rows-1,1):
        for column in range(1,columns-1,1):
            sum_x = sobel_sum_x(row,column)
            sum_y = sobel_sum_y(row,column)
            value = int(math.sqrt(sum_x*sum_x + sum_y*sum_y))
            if value>255:
                value = 255
            image3[row][column] = value
    image_name = image_name.split('.')
    disk_name = image_name[0] + "sobel.jpg"
    cv2.imshow("Original Image",image2)
    cv2.imshow("Sobel Image",image3)
    cv2.imwrite(disk_name,image3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Laplacian(image_name):
#    image_name = raw_input()
    image = cv2.imread(image_name,-1)
    imagegray = cv2.imread(image_name,-1)
    imagegray2 = cv2.imread(image_name,-1)
    imagegray3 = cv2.imread(image_name, -1)
    if type(image[0][0]) is numpy.ndarray:
        imagegray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        imagegray2 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        imagegray3 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    rows = len(imagegray)
    columns = len(imagegray[1])

#    for row in range(rows):
#        for column in range(columns):
#            emptyrow += [0]
#        emptyimage += [emptyrow]
#


    all_values = []       
    def calc(xindex,yindex):
        total = int(imagegray[xindex][yindex-1]) + int(imagegray[xindex][yindex+1]) + int(imagegray[xindex-1][yindex]) + int(imagegray[xindex+1][yindex]) - 4*int(imagegray[xindex][yindex])
        return total
        
    for row in range(1,rows-1,1):
        for column in range(1,columns-1,1):
            value = calc(row,column)
            all_values += [value]
            if value < 0:
                value = 0
            elif value > 255:
                value = 255
            imagegray2[row][column] = value
    
    maximum = max(all_values)
    minimum = min(all_values)
    denom = maximum - minimum
    for row in range(1,rows-1,1):
        for column in range(1,columns-1,1):
            current_val = imagegray2[row][column]
            numer = current_val - minimum
            replace = (numer * 255)/denom
            imagegray3[row][column] = replace
    image_name = image_name.split('.')
    disk_name = image_name[0] + "laplacian.jpg"
    cv2.imshow("OLD",imagegray)
    cv2.imshow("Laplacian",imagegray3)
    cv2.imwrite(disk_name,imagegray3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def connected_component(image_name):
    image = cv2.imread(image_name,0)
    imagenew = cv2.imread(image_name)
    threshold = 128
    rows = len(image)
    columns = len(image[1])
    
    for row in range(rows):
        for column in range(columns):
            if image[row][column] >= threshold:
                image[row][column] = 255
            else:
                image[row][column] = 0                
    emptyimage = []
    emptyrow = []
    for row in range(rows):
        for column in range(columns):
            emptyrow += [0]
        emptyimage += [emptyrow]
        emptyrow = []
    equiv = {}
    counter = 1    
    for row in range(rows):
        for column in range(columns):
            if image[row][column] == 255:
                if row - 1 >= 0:
                    top = emptyimage[row-1][column]
                else:
                    top = 0
                if column - 1 >=0:
                    left = emptyimage[row][column-1]
                else:
                    left = 0
                if row - 1 < 0:
                    if column - 1 < 0:
                        emptyimage[row][column] = counter
                    else:
                        emptyimage[row][column] = left
                elif column - 1 < 0:
                    emptyimage[row][column] = top
                else:
                    if top == 0 and left == 0:
                        counter += 1
                        emptyimage[row][column] = counter
                    elif top == 0:
                        emptyimage[row][column]  = left
                    elif left == 0:
                        emptyimage[row][column] = top
                    else:
                        emptyimage[row][column] = left
                        if left != top:
                            if top not in equiv:
                                equiv[top] = left
    total = 0
    visited = []
    start = 2
    connected = {}
    
    while start <= counter:
        if start not in visited:
            visited += [start]
            connected[start] = []
            current = start
            while current in equiv:
                current = equiv[current]
                if current not in visited:
                    visited += [current]
                    connected[start] += [current]
            start += 1
        else:
            start += 1
    


#    print connected

    range_red = 255
    range_green = 255
    range_blue = 255    
    colormap = {}
    
    for values in connected:
        red = random.randint(0,range_red)
        green = random.randint(0,range_green)
        blue = random.randint(0,range_blue)
        colormap[values] = [blue,green,red]
        
#    print colormap
#    print
#    print
    mapper = {}   
    for values in connected:
        mapper[values] = values
        
    for values in connected:
        for points in connected[values]:
            mapper[points] = values
                
 #   print mapper
    for row in range(rows):
        for column in range(columns):
            if emptyimage[row][column] == 0:
                pass
            else:
                if emptyimage[row][column] in mapper:
                    emptyimage[row][column] = mapper[emptyimage[row][column]]
    areas = {}
    for row in range(rows):
        for column in range(columns):
            if emptyimage[row][column] == 0:
                pass
            else:
                value = emptyimage[row][column]
                if value in colormap:
                    imagenew[row][column] = colormap[value]
                if value in mapper:
                    if value in areas:
                        areas[value] += 1
                    else:
                        areas[value] = 1
    comps = 0
    for component in connected:
        comps += 1
        if component in areas:
            print "Area of component",comps,areas[component]
    cv2.imshow("Labeled",imagenew)
    cv2.imshow("Original",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
            
def IPyramid(imagename):
    image1 = cv2.imread(imagename,-1)
    imagebw = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    print type(imagebw)
    def level(xstart,xend,ystart,yend):
        halfx = (xend - xstart + 1)/2
        halfy = (yend - ystart + 1)/2
        newimage = []
        emptyrow = []
        for row in range(halfx):
            for col in range(halfy):
                emptyrow += [(0)]
            newimage += [emptyrow]
            emptyrow = []
        return newimage
    def replace(image,diskname):
        xstart = 0
        xend = len(image)-1
        ystart = 0
        yend = len(image[0])-1
        resized = cv2.resize(image, (0,0), fx=0.5, fy=0.5) 
#        halfx = (xend - xstart + 1)/2
#        halfy = (yend - ystart + 1)/2
        #emptyimage = level(xstart,xend,ystart,yend)
        #emptyimage = numpy.zeros((halfx,halfy))
        xindex = 0
        yindex = 0
        for row in range(xstart,xend,2):
            for col in range(ystart,yend,2):
                sum_total = int(image[row][col]) + int(image[row][col+1]) + int(image[row+1][col]) + int(image[row+1,col+1])
                sum_total = sum_total/4
                resized[xindex][yindex] = int(sum_total)
                yindex += 1
            xindex += 1
            yindex = 0
        print type(resized)
        cv2.imwrite(diskname,resized)
        return resized
    lvl = 2
    total_lvl = 6
    name =  "level"
    initial = imagebw
    imgname = imagename.split(".")
    cv2.imwrite(imgname[0]+name+".jpg",initial)
    while lvl <= total_lvl:
        diskname = imgname[0]+name + str(lvl) +".jpg"
        new = replace(initial,diskname)   
        initial = new
        lvl += 1

def Scalingup(imname):
    image = cv2.imread(imname,-1)
    rows = len(image)
    columns = len(image[0])
    #scaledup_comp = cv2.resize(image,(0,0),fx = 2,fy = 2)
    #newscaledup = cv2.resize(image,(0,0),fx = 2,fy = 2)
#    scaledup = []
#    dupscaledup = []
#    emptycol = []
    rowmax = 2*rows + 1
    colmax = 2*columns + 1
    nscaling = numpy.zeros((rowmax,colmax),numpy.uint8)
    nscaling2 = numpy.zeros((rowmax,colmax),numpy.uint8)
#    for row in range(rowmax):
#        for column in range(colmax):
#            emptycol += [0]
#        scaledup += [emptycol]
#        emptycol = []
#    print scaledup
#    print scaledup_comp
    for row in range(rows):
        for column in range(columns):
            currow = 2*row+1
            curcol = 2*column+1
            nscaling[currow][curcol] = image[row][column]
#    print scaledup_comp
#    print scaledup
    for row in range(rowmax-1):
        for column in range(colmax-1):
            value = int(nscaling[row][column])+int(nscaling[row+1][column])+int(nscaling[row][column+1])+int(nscaling[row+1][column+1])
            nscaling2[row][column] = value
    for row in range(rowmax-1):
        nscaling2[row][colmax-1] = int(nscaling2[row][colmax-2])
    for column in range(colmax):
        nscaling2[rowmax-1][column] = int(nscaling2[rowmax-2][column])
    imgname = imname.split(".")
    cv2.imwrite(imgname[0]+"0"+".jpg",nscaling2)
    cv2.imshow("Scaledup",nscaling2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()   
    
def scalingup(imname):
    image = cv2.imread(imname,-1)
    rows = len(image)
    columns = len(image[0])
    #scaledup_comp = cv2.resize(image,(0,0),fx = 2,fy = 2)
    #newscaledup = cv2.resize(image,(0,0),fx = 2,fy = 2)
#    scaledup = []
#    dupscaledup = []
#    emptycol = []
    rowmax = 2*rows
    colmax = 2*columns
    nscaling = numpy.zeros((rowmax,colmax),numpy.uint8)
    nscaling2 = numpy.zeros((rowmax,colmax),numpy.uint8)
#    for row in range(rowmax):
#        for column in range(colmax):
#            emptycol += [0]
#        scaledup += [emptycol]
#        emptycol = []
#    print scaledup
#    print scaledup_comp
    for row in range(rows):
        for column in range(columns):
            currow = 2*row
            curcol = 2*column
            nscaling[currow][curcol] = image[row][column]

    for row in range(rows):
        for column in range(columns-1):
            currow = 2*row
            curcol = 2*column+1
            value = (int(nscaling[currow][curcol-1]) + int(nscaling[currow][curcol+1]))/2 + 1
            if value > 255:
                value = 255
            elif value < 0:
                value = 0
            nscaling[currow][curcol] = value

    for row in range(rows-1):
        for column in range(colmax-1):
            currow = 2*row+1
            curcol = column
            value = (int(nscaling[currow-1][curcol])+int(nscaling[currow+1][curcol]))/2 + 1
            if value > 255:
                value = 255
            elif value < 0:
                value = 0
            nscaling[currow][curcol] = value
    for row in range(rowmax-1):
        nscaling[row][colmax-1] = int(nscaling[row][colmax-2])
    for column in range(colmax):
        nscaling[rowmax-1][column] = int(nscaling[rowmax-2][column])
    imgname = imname.split(".")
    diskname = imgname[0]+"1"+".jpg"
    cv2.imwrite(diskname,nscaling)
    cv2.imshow("Scaledup",nscaling)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
class MyGUI:
    def __init__(self,master):
        self.master = master
        master.title("Project 1")

        #self.label = Label(master,text = "Button1",padx = 10)
        #self.label.grid(row = 1)
        #Label for Noise Generator
        self.titlenoise = TT.Label(master,text = "Noise ",padx = 10,pady = 10)
        self.titlenoise.grid(row = 1,column = 0)

        #Textbox for Noise Image
        self.textbox = TT.Entry(master)
        self.textbox.grid(row = 1,column = 1)

        #Label for White pixel percentage
        self.titlenoisewhite = TT.Label(master,text = "White Pixel Percentage ",padx = 10,pady = 10)
        self.titlenoisewhite.grid(row = 1,column = 2)

        #Textbox for White Pixel percentage
        self.textboxwhite = TT.Entry(master)
        self.textboxwhite.grid(row = 1,column = 3)

        #Label for black pixel percentage
        self.titlenoiseblack = TT.Label(master,text = "Black Pixel percentage ",padx = 10,pady = 10)
        self.titlenoiseblack.grid(row = 1,column = 4)

        #Textbox for black pixel percentage
        self.textboxblack = TT.Entry(master)
        self.textboxblack.grid(row = 1,column = 5)

        #Button for Noise Generator
        self.button = TT.Button(master, text = "Noise",padx = 10,pady = 10,command = self.sample)
        self.button.grid(row = 1,column = 8)

        #Label for Ptile Generator
        self.titlePtile = TT.Label(master, text = "Ptile Thresholding",padx = 10,pady = 10)
        self.titlePtile.grid(row = 2,column = 0)

        #Label for Ptile Image
        self.textboxPtile = TT.Entry(master)
        self.textboxPtile.grid(row = 2,column = 1)

        #Label for PValue
        self.titlePvalue = TT.Label(master, text = "Pvalue",padx = 10,pady = 10)
        self.titlePvalue.grid(row = 2,column = 2)

        #Label for Pvalue Entry
        self.textboxPvalue = TT.Entry(master)
        self.textboxPvalue.grid(row = 2,column = 3)

        #Label for Ptile Button
        self.PtileButton = TT.Button(master,text = "Ptile",padx = 10,command = self.ptilehelper)
        self.PtileButton.grid(row = 2,column = 8)

        #Label for Binary Threshold
        self.titleBT = TT.Label(master, text = "Binary Threshold",padx = 10,pady=10)
        self.titleBT.grid(row = 3,column = 0)

        #Label for Binary Threshold Image
        self.textboxBTI = TT.Entry(master)
        self.textboxBTI.grid(row = 3,column = 1)

        #Label for Binary Threshold Value
        self.titleBTLabel = TT.Label(master, text = "Threshold",padx = 10,pady = 10)
        self.titleBTLabel.grid(row = 3,column = 2)

        #Binary Threshold Value
        self.textboxBTValue = TT.Entry(master)
        self.textboxBTValue.grid(row = 3,column = 3)

        #Binary Threshold Button
        self.BTButton = TT.Button(master,text = "Binary Threshold",padx = 10,command = self.binaryhelper)
        self.BTButton.grid(row = 3,column = 8)

        #Label for Averaging
        self.titleAvgLabel = TT.Label(master, text = "Average Filtering",padx = 10,pady = 10)
        self.titleAvgLabel.grid(row = 4,column = 0)

        #Image for Averaging
        self.textAvgIm = TT.Entry(master)
        self.textAvgIm.grid(row = 4,column = 1)

        #Label for Mask Size
        self.titleMaskLabel = TT.Label(master, text = "Mask Size",padx = 10,pady = 10)
        self.titleMaskLabel.grid(row = 4,column = 2)

        #Mask Size
        self.MaskSize = TT.Entry(master)
        self.MaskSize.grid(row = 4,column = 3)

        #Label for white pixel noise
        self.titleavgwhite = TT.Label(master,text = "White Pixel % ",padx = 10,pady = 10)
        self.titleavgwhite.grid(row = 4,column = 4)

        #Entry for white pixel noise
        self.Avgwhitepixel = TT.Entry(master)
        self.Avgwhitepixel.grid(row = 4,column = 5)

        #Label for black pixel noise
        self.titleavgblack = TT.Label(master,text = "Black Pixel % ",padx = 10,pady = 10)
        self.titleavgblack.grid(row = 4,column = 6)

        #Entry for black pixel noise
        self.Avgblackpixel = TT.Entry(master)
        self.Avgblackpixel.grid(row = 4,column = 7)

        #Button for Averaging
        self.AvgButton = TT.Button(master,text = "Average Image",padx = 10,command = self.averagehelper)
        self.AvgButton.grid(row = 4,column = 8)

        #Label for Median Filtering
        self.titleMedLabel = TT.Label(master, text = "Median Filtering",padx = 10,pady = 10)
        self.titleMedLabel.grid(row = 5,column = 0)

        #Image for Median Filtering
        self.textMedIm = TT.Entry(master)
        self.textMedIm.grid(row = 5,column = 1)

        #Label for Mask Size
        self.titleMaskMedLabel = TT.Label(master, text = "Mask Size",padx = 10,pady = 10)
        self.titleMaskMedLabel.grid(row = 5,column = 2)

        #Mask Size
        self.MaskSizeMed = TT.Entry(master)
        self.MaskSizeMed.grid(row = 5,column = 3)

         #Label for white pixel noise for median
        self.titlemedwhite = TT.Label(master,text = "White Pixel % ",padx = 10,pady = 10)
        self.titlemedwhite.grid(row = 5,column = 4)

        #Entry for white pixel noise for median
        self.Medwhitepixel = TT.Entry(master)
        self.Medwhitepixel.grid(row = 5,column = 5)

        #Label for black pixel noise for median
        self.titlemedblack = TT.Label(master,text = "Black Pixel % ",padx = 10,pady = 10)
        self.titlemedblack.grid(row = 5,column = 6)

        #Entry for black pixel noise for median
        self.Medblackpixel = TT.Entry(master)
        self.Medblackpixel.grid(row = 5,column = 7)

        #Button for Median
        self.MedButton = TT.Button(master,text = "Median Filter",padx = 10,command = self.medianhelper)
        self.MedButton.grid(row = 5,column = 8)

        #Label for Median Filtering
        self.titleImageEnh = TT.Label(master, text = "Image Enhancement",padx = 10,pady = 10)
        self.titleImageEnh.grid(row = 6,column = 0)

        #Image for Median Filtering
        self.textImageEnh = TT.Entry(master)
        self.textImageEnh.grid(row = 6,column = 1)

        self.ImageEnhButton = TT.Button(master,text = "Image Enhancement",padx = 10,command = self.ImageEnhhelper)
        self.ImageEnhButton.grid(row = 6, column = 8)

        self.titleIterLabel = TT.Label(master, text = "Iterative Filtering",padx = 10,pady = 10)
        self.titleIterLabel.grid(row = 7,column = 0)

         #Image for Median Filtering
        self.textIterIm = TT.Entry(master)
        self.textIterIm.grid(row = 7,column = 1)

        self.MedButton = TT.Button(master,text = "Iterative Filtering",padx = 10,command = self.iterative_threshold)
        self.MedButton.grid(row = 7,column = 8)

        self.titleRobert = TT.Label(master,text = "Robert's Cross",padx = 10,pady = 10)
        self.titleRobert.grid(row = 8, column = 0)

        self.textRobert = TT.Entry(master)
        self.textRobert.grid(row = 8, column = 1)

        self.buttonRobert = TT.Button(master,text = "Robert's Cross",command = self.Roberthelper)
        self.buttonRobert.grid(row = 8,column = 8)

        self.titleKirsch = TT.Label(master,text = "Kirsch Operator",padx = 10,pady = 10)
        self.titleKirsch.grid(row = 9, column = 0)

        self.textKirsch = TT.Entry(master)
        self.textKirsch.grid(row = 9, column = 1)

        self.buttonKirsch = TT.Button(master,text = "Kirsch Operator",command = self.Kirschhelper)
        self.buttonKirsch.grid(row = 9,column = 8)

        self.titleSobel = TT.Label(master,text = "Sobel Operator",padx = 10,pady = 10)
        self.titleSobel.grid(row = 10, column = 0)

        self.textSobel = TT.Entry(master)
        self.textSobel.grid(row = 10, column = 1)

        self.buttonSobel = TT.Button(master,text = "Sobel Operator",command = self.Sobelhelper)
        self.buttonSobel.grid(row = 10,column = 8)

        self.titleLaplacian = TT.Label(master,text = "Laplacian Operator",padx = 10,pady = 10)
        self.titleLaplacian.grid(row = 11, column = 0)

        self.textLaplacian = TT.Entry(master)
        self.textLaplacian.grid(row = 11, column = 1)

        self.buttonLaplacian = TT.Button(master,text = "Laplacian Operator",command = self.Laplacianhelper)
        self.buttonLaplacian.grid(row = 11,column = 8)

        self.titleComps = TT.Label(master,text = "Component Identifier",padx = 10,pady = 10)
        self.titleComps.grid(row = 12, column = 0)

        self.textComps = TT.Entry(master)
        self.textComps.grid(row = 12, column = 1)

        self.buttonComps = TT.Button(master,text = "Component Identifier",command = self.Componenthelper)
        self.buttonComps.grid(row = 12,column = 8)
        
        self.titleIPyramid = TT.Label(master,text = "Image Pyramid",padx = 10,pady = 10)
        self.titleIPyramid.grid(row = 13, column = 0)
        
        self.textIPyramid = TT.Entry(master)
        self.textIPyramid.grid(row = 13, column = 1)
        
        self.buttonIPyramid = TT.Button(master,text = "Image Pyramid",command = self.IPyramidhelper)
        self.buttonIPyramid.grid(row = 13,column = 8)
        
        self.titleIScale = TT.Label(master,text = "Image ScaleUp",padx = 10,pady = 10)
        self.titleIScale.grid(row = 14, column = 0)
        
        self.textIScale = TT.Entry(master)
        self.textIScale.grid(row = 14, column = 1)
        
        self.buttonIScale = TT.Button(master,text = "Scaleup",command = self.IScaleUp)
        self.buttonIScale.grid(row = 14,column = 8)
        
    def sample(self):
        text_value = self.textbox.get()
        white = self.textboxwhite.get()
        black = self.textboxblack.get()
        noise(text_value,int(white),int(black))

    def ptilehelper(self):
        text_value = self.textboxPtile.get()
        Pvalue = self.textboxPvalue.get()
        PTile(text_value,Pvalue)

    def binaryhelper(self):
        text_value = self.textboxBTI.get()
        threshold = self.textboxBTValue.get()
        toBinary(text_value,threshold)

    def averagehelper(self):
        text_value = self.textAvgIm.get()
        mask_size = self.MaskSize.get()
        white_percent = self.Avgwhitepixel.get()
        black_percent = self.Avgblackpixel.get()
        averaging(text_value,mask_size,int(white_percent),int(black_percent))

    def medianhelper(self):
        text_value = self.textMedIm.get()
        mask_size = self.MaskSizeMed.get()
        white_percent = self.Medwhitepixel.get()
        black_percent = self.Medblackpixel.get()
        medianfilter(text_value,mask_size,int(white_percent),int(black_percent))

    def ImageEnhhelper(self):
        text_value = self.textImageEnh.get()
        enhance(text_value)

    def iterative_threshold(self):
        text_value = self.textIterIm.get()
        IterativeThresholding(text_value)

    def Roberthelper(self):
        text_value = self.textRobert.get()
        Roberts(text_value)

    def Kirschhelper(self):
        text_value = self.textKirsch.get()
        Kirsch(text_value)

    def Sobelhelper(self):
        text_value = self.textSobel.get()
        Sobel(text_value)

    def Laplacianhelper(self):
        text_value = self.textLaplacian.get()
        Laplacian(text_value)
        
    def Componenthelper(self):
        text_value = self.textComps.get()
        connected_component(text_value)
        
    def IPyramidhelper(self):
        text_value = self.textIPyramid.get()
        IPyramid(text_value)
        
    def IScaleUp(self):
        text_value = self.textIScale.get()
        Scalingup(text_value)

root = TT.Tk()
mygui = MyGUI(root)
root.mainloop()
