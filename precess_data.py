# encoding: utf-8
'''
@author: victoria
@file: precess_data.py
@time: 2018/4/25 16:56
@desc: 图片的预处理类
'''

import cv2
import os
import re


def eachFile(folder):
    allFile = os.listdir(folder)
    fileNames = []
    for file in allFile:
        fullPath = os.path.join(folder, file)
        fileNames.append(fullPath)

    return fileNames


def changeFileName(oldFileName, newFileName):
    try:
        os.rename(oldFileName, newFileName)
    except:
        print('an error occurred')


def deleteFile(filePath):
    if os.path.isfile(filePath):
        try:
            os.remove(filePath)
        except:
            print('remove error')


class DataProcessor(object):
    def __init__(self):
        pass

    #缩放图片
    def reduce_size(self, srcFolder, destFolder, magnification):
        if (magnification >= 1):
            print('Magnification must be less than 1！')
            return False

        fileNames = eachFile(srcFolder)
        for fileName in fileNames:
            image = cv2.imread(fileName)

            # 删除不可用图片
            if(image is None):
                deleteFile(fileName)
                continue

            width, height, channel = image.shape
            # 向下取整缩小
            output = cv2.resize(image, (int(width * magnification), int(height * magnification)),
                                interpolation=cv2.INTER_AREA)

            fileName = re.findall(srcFolder+'/(.*)' ,fileName)[0]
            newFilePath = os.path.join(destFolder, fileName)
            cv2.imwrite(newFilePath, output)


    def restore_size(self, srcFolder, destFolder, referFolder):
        fileNames = eachFile(srcFolder)
        for fileName in fileNames:
            referFileName = re.findall(srcFolder + '/(.*)', fileName)[0]
            referFilePath = os.path.join(referFolder, referFileName)
            referImage = cv2.imread(referFilePath)

            if(referImage is None):
                print(referFilePath)
                # continue

            width, height, channel = referImage.shape
            # reshape to original size
            src_image = cv2.imread(fileName)
            # height and width should reverse, and I don't not why
            output = cv2.resize(src_image, (height, width),
                                interpolation=cv2.INTER_CUBIC)

            fileName = re.findall(srcFolder + '/(.*)', fileName)[0]
            newFilePath = os.path.join(destFolder, fileName)
            cv2.imwrite(newFilePath, output)


    #切割图片
    def cut_images(self, srcFolder, destFolder):
        num = 0
        fileNames = eachFile(srcFolder)
        for fileName in fileNames:
            image = cv2.imread(fileName)

            if(image is None):
                deleteFile(fileName)
                continue

            width, height, channel = image.shape
            if(width > 400 and height > 400):
                num += 1
                print(num, width, height)
                image = cv2.resize(image, (400, 400))  #图片缩放至400*400
                fileName = re.findall(srcFolder + '/(.*)', fileName)[0]
                newFilePath = os.path.join(destFolder, fileName)
                cv2.imwrite(newFilePath, image)  #写入图片





if __name__ == '__main__':
    # utils.deleteGif('./images/origin')
    dataProcessor = DataProcessor()
    # dataProcessor.cut_images('./images/origin', './images/cut')
    dataProcessor.reduce_size('./data/faces/cut', './data/faces/lowResolution', 1/4)  #压缩
    dataProcessor.restore_size('./data/faces/lowResolution', './data/faces/interpolation','./data/faces/cut') #双插值放大
    # dataProcessor.saveToH5File('./images/cut', 50, './dataset.h5')

