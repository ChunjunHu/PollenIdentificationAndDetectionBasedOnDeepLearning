import os
import time
from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage import io
#io.use_plugin('matplotlib')
import cv2
import numpy as np
from skimage import segmentation
from skimage.segmentation import felzenszwalb
import copy
import torch
import torch.nn as nn
from test1 import predictProcess



class Args(object):
    input_image_path = 'image/tridax_02.jpg'
    train_epoch = 100
    mod_dim1 = 64  #
    mod_dim2 = 32
    gpu_id = 0

    min_label_num = 3  # if the label number small than it, break loop
    max_label_num = 256  # if the label number small than it, start to show result image.


class MyNet(nn.Module):
    def __init__(self, inp_dim, mod_dim1, mod_dim2):
        super(MyNet, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(inp_dim, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mod_dim2),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim2, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mod_dim2),
        )

    def forward(self, x):
        return self.seq(x)




# 两个检测框框是否有交叉，如果有交集则返回重叠度 IOU, 如果没有交集则返回 0
def bb_overlab(x1, y1, w1, h1, x2, y2, w2, h2):
    '''
    说明：图像中，从左往右是 x 轴（0~无穷大），从上往下是 y 轴（0~无穷大），从左往右是宽度 w ，从上往下是高度 h
    :param x1: 第一个框的左上角 x 坐标
    :param y1: 第一个框的左上角 y 坐标
    :param w1: 第一幅图中的检测框的宽度
    :param h1: 第一幅图中的检测框的高度
    :param x2: 第二个框的左上角 x 坐标
    :param y2:
    :param w2:
    :param h2:
    :return: 两个如果有交集则返回重叠度 IOU, 如果没有交集则返回 0
    '''
    areaFlag = 0
    if(x1>x2+w2):
        return 0
    if(y1>y2+h2):
        return 0
    if(x1+w1<x2):
        return 0
    if(y1+h1<y2):
        return 0
    colInt = abs(min(x1 +w1 ,x2+w2) - max(x1, x2))
    rowInt = abs(min(y1 + h1, y2 +h2) - max(y1, y2))
    overlap_area = colInt * rowInt
    area1 = w1 * h1
    area2 = w2 * h2
    overlapRate = overlap_area / (area1 + area2 - overlap_area)
    if area1 >= area2:
        selfOverlapRate = overlap_area /  area2
    if area1 < area2:
        selfOverlapRate = overlap_area / area1
    if area1 == area2 and selfOverlapRate > 0.8 :
        return 4
    elif overlapRate > 0.1 and selfOverlapRate > 0.8 :
        # if area1 > area2:
        return 5
    else:
        return 0
        # if area1 < area2:
        #     return 1

def changeAxis(maxX, minX, maxY, minY):
    w = maxX - minX
    h = maxY - minY
    x = minX
    y = minY
    return x, y, w, h

def calcualteArea(w, h):
    area = w * h
    return area

def loopCalculate(areaDictCopy, resultPosition,changeObjectsPosition,objectsPosition):
    maxRepeat = 0
    length = 0
    areaDictDeepCopy = copy.deepcopy(areaDictCopy)
    length = len(areaDictCopy)
    for k in areaDictDeepCopy.keys():
        maxKey = max(areaDictCopy,key=areaDictCopy.get)
        if maxKey != k:
            x1, y1, w1, h1 = changeObjectsPosition[maxKey]
            x2, y2, w2, h2 = changeObjectsPosition[k]
            returnResult = bb_overlab(x1, y1, w1, h1, x2, y2, w2, h2)
            if returnResult == 4:
                maxX1, minX1, maxY1, minY1 = objectsPosition[maxKey]
                maxX2, minX2, maxY2, minY2 = objectsPosition[k]
                if maxX1 >= maxX2:
                    maxX1 = maxX1
                if maxX1 < maxX2:
                    maxX1 = maxX2
                if minX1 <= minX2:
                    minX1 = minX1
                if minX1 > minX2:
                    minX1 = minX2
                if maxY1 >= maxY2:
                    maxY1 = maxY1
                if maxY1 < maxY2:
                    maxY1 = maxY2
                if minY1 <= minY2:
                    minY1 = minY1
                if minY1 > minY2:
                    minY1 = minY2
                resultPosition.append([maxX1, minX1, maxY1, minY1])
                maxRepeat = 1
                del [areaDictCopy[k]]
            elif returnResult == 5:
                del [areaDictCopy[k]]
        ss = length - k
        if ss == 1:
            if maxRepeat == 0:
                resultPosition.append(objectsPosition[maxKey])

            del [areaDictCopy[maxKey]]
            del areaDictDeepCopy
            loopCalculate(areaDictCopy, resultPosition,changeObjectsPosition,objectsPosition)


def run():
    start_time0 = time.time()

    args = Args()
    torch.cuda.manual_seed_all(1943)
    np.random.seed(1943)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)  # choose GPU:0
    image = cv2.imread(args.input_image_path)

    '''segmentation ML'''
    seg_map = felzenszwalb(image, scale=32, sigma=0.5, min_size=64)
    # seg_map = segmentation.slic(image, n_segments=10000, compactness=100)
    seg_map = seg_map.flatten()
    seg_lab = [np.where(seg_map == u_label)[0]
               for u_label in np.unique(seg_map)]

    '''train init'''
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    tensor = image.transpose((2, 0, 1))
    tensor = tensor.astype(np.float32) / 255.0
    tensor = tensor[np.newaxis, :, :, :]
    tensor = torch.from_numpy(tensor).to(device)

    model = MyNet(inp_dim=3, mod_dim1=args.mod_dim1, mod_dim2=args.mod_dim2).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-1, momentum=0.0)

    image_flatten = image.reshape((-1, 3))
    color_avg = np.random.randint(255, size=(args.max_label_num, 3))
    show = image

    '''train loop'''
    start_time1 = time.time()
    model.train()
    for batch_idx in range(args.train_epoch):
        '''forward'''
        optimizer.zero_grad()
        output = model(tensor)[0]
        output = output.permute(1, 2, 0).view(-1, args.mod_dim2)
        target = torch.argmax(output, 1)
        im_target = target.data.cpu().numpy()

        '''refine'''
        for inds in seg_lab:
            u_labels, hist = np.unique(im_target[inds], return_counts=True)
            im_target[inds] = u_labels[np.argmax(hist)]

        '''backward'''
        target = torch.from_numpy(im_target)
        target = target.to(device)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        '''show image'''
        un_label, lab_inverse = np.unique(im_target, return_inverse=True, )
        if un_label.shape[0] < args.max_label_num:  # update show
            img_flatten = image_flatten.copy()
            if len(color_avg) != un_label.shape[0]:
                color_avg = [np.mean(img_flatten[im_target == label], axis=0, dtype=np.int) for label in un_label]
            for lab_id, color in enumerate(color_avg):
                img_flatten[lab_inverse == lab_id] = color
            show = img_flatten.reshape(image.shape)
        cv2.imshow("seg_pt", show)
        cv2.waitKey(1)

        print('Loss:', batch_idx, loss.item())
        if len(un_label) < args.min_label_num:
            break
    print(img_flatten)
    print(len(img_flatten))


    '''save'''
    time0 = time.time() - start_time0
    time1 = time.time() - start_time1
    print('PyTorchInit: %.2f\nTimeUsed: %.2f' % (time0, time1))
    cv2.imwrite("seg_%s_%ds.jpg" % (args.input_image_path[6:-4], time1), show)

    return img_flatten, image



if __name__ == '__main__':

    objectsPosition = []
    #colorDataBase = []
    flag = [0,0,0,0]
    areaDict = {0: 0}
    
    colorLable, imagePara = run()
    backGround = colorLable[0]
    backGround = backGround.tolist()
    print(backGround)
    vertail, horizontal = imagePara.shape[0], imagePara.shape[1]
    ColorLable = []
    # for i in range(len(colorLable)):
    #     if colorDataBase == None:
    #         colorDataBase[i] = colorLable[i]
    #     else:
    #         for j in colorLable:
    print('去重方案2:\n',np.array(list(set([tuple(t) for t in colorLable]))))     
    colorDataBase = np.array(list(set([tuple(t) for t in colorLable])))
    colorLable = colorLable.tolist()
    colorDataBase = colorDataBase.tolist()
    for i in range(len(colorLable)):
        for j in range(len(colorDataBase)):
            # print("------------------")
            # print(colorLable[i])
            # print("----------------")
            # print(colorDataBase[j])
            if colorLable[i] == colorDataBase[j]:
                ColorLable.append(j)
                #colorLable = np.delete(colorLable, i, 0)
                #colorLable = np.insert(colorLable, i, j, 0)
                # del colorLable[i]
                # colorLable = colorLable.insert(i, j)
                # print(colorLable[i])
                #colorLable.insert(i, j)
                #colorLable[i] = j

        
    print(vertail)
    print(horizontal)
    print(ColorLable)


    #print(colorLable[0][0])
    ColorLableName, colorLableInverse = np.unique(ColorLable, return_inverse = True)
    colorLableName = [[] for i in range(len(ColorLableName))]
    resultPosition = []
    resultCoding = []
    for i in range(len(ColorLableName)):
        colorLableName[i] = colorDataBase[ColorLableName[i]]
    objectsPosition = [[0 for col in range(4)] for row in range(len(colorLableName))]
    print(colorLableName)
    print(len(colorLableInverse))
    for i in range(len(colorLableInverse)):
        
        lableNamePosition = colorLableInverse[i]
        lableName = colorLableName[lableNamePosition]
        if lableName != backGround:
            LableNamePosition = lableNamePosition
            maxY = int(i / horizontal)
            maxX = int(i % horizontal)
            minY = int(maxY)
            minX = int(maxX)
            #print(maxX, minX, maxY, minY)
            if objectsPosition[LableNamePosition] == flag:
                objectsPosition[LableNamePosition] = [maxX, minX, maxY, minY]
            else:
                if maxX > objectsPosition[LableNamePosition][0]:
                    objectsPosition[LableNamePosition][0] = maxX
                if minX < objectsPosition[LableNamePosition][1]:
                    objectsPosition[LableNamePosition][1] = minX
                if maxY > objectsPosition[LableNamePosition][2]:
                    objectsPosition[LableNamePosition][2] = maxY
                if minY < objectsPosition[LableNamePosition][3]:
                    objectsPosition[LableNamePosition][3] = minY
            #print(objectsPosition[LableNamePosition])
    print(objectsPosition)
    list1 = copy.deepcopy(objectsPosition)
    for i in range(len(list1)):
        if list1[i] == flag:
            objectsPosition.remove(flag)
    print(objectsPosition)
    changeObjectsPosition = copy.deepcopy(objectsPosition)
    origionalPosition = [0, 0, 0, 0]
    # for i in range(len(objectsPosition)):
    #     cv2.rectangle(imagePara, (objectsPosition[i][1], objectsPosition[i][3]), (objectsPosition[i][0], objectsPosition[i][2]),(0,255,0),3)
    for i in range(len(objectsPosition)):
        
        origionalPosition = objectsPosition[i]
        x, y, w, h = changeAxis(origionalPosition[0], origionalPosition[1], origionalPosition[2], origionalPosition[3])
        changeObjectsPosition[i] = [x, y, w, h]
    for i in range(len(changeObjectsPosition)):
        area = calcualteArea(changeObjectsPosition[i][2], changeObjectsPosition[i][3])
        areaDict[i] = area
        print(areaDict)
    areaDictCopy = copy.deepcopy(areaDict)
    print(areaDictCopy)
    loopCalculate(areaDictCopy, resultPosition,changeObjectsPosition,objectsPosition)
    # for k in areaDict.keys():
    #     maxKey = max(areaDictCopy,key=areaDictCopy.get)
    #     if maxKey != k:
    #         x1, y1, w1, h1 = changeObjectsPosition[maxKey]
    #         x2, y2, w2, h2 = changeObjectsPosition[k]
    #         returnResult = bb_overlab(x1, y1, w1, h1, x2, y2, w2, h2)
    #         if returnResult == 0:
    #             maxX1, minX1, maxY1, minY1 = objectsPosition[maxKey]
    #             maxX2, minX2, maxY2, minY2 = objectsPosition[k]
    #             if maxX1 >= maxX2:
    #                 maxX1 = maxX1
    #             if maxX1 < maxX2:
    #                 maxX1 = maxX2
    #             if minX1 <= minX2:
    #                 minX1 = minX1
    #             if minX1 > minX2:
    #                 minX1 = minX2
    #             if maxY1 >= maxY2:
    #                 maxY1 = maxY1
    #             if maxY1 < maxY2:
    #                 maxY1 = maxY2
    #             if minY1 <= minY2:
    #                 minY1 = minY1
    #             if minY1 > minY2:
    #                 minY1 = minY2
    #             resultPosition.append([maxX1, minX1, maxY1, minY1])
    #             del [areaDictCopy[maxKey]]
    #             del [areaDictCopy[k]]
    #         elif returnResult == 1:
    #             del [areaDictCopy[k]]
    # for q in areaDictCopy.keys():
    #     po = objectsPosition[q]
    #     resultPosition.append(po)
    print(resultPosition)
    resultPositionCopy = copy.deepcopy(resultPosition)
    # for i in range(len(resultPosition)):
    #     cv2.rectangle(imagePara, (resultPosition[i][1], resultPosition[i][3]), (resultPosition[i][0], resultPosition[i][2]),(0,255,0),3)
    # cv2.imshow("Result", imagePara)
    # cv2.waitKey(1)
    # cv2.imwrite("segs.jpg", imagePara)

    for i in range(len(resultPositionCopy)):
        x1, x2, y1, y2 = resultPositionCopy[i]
        x1 = x1 + 10
        x2 = x2 - 10
        y1 = y1 + 10
        y2 = y2 - 10
        if x1 > horizontal:
            x1 = horizontal
        if y1 > vertail:
            y1 = vertail
        if x2 < 0:
            x2 = 0
        if y2 < 0:
            y2 = 0
        resultPositionCopy[i] = x1, x2, y1, y2
    print(resultPositionCopy)

    for i in range(len(resultPositionCopy)):
        imgCopy = imagePara.copy()
        cropImage = imgCopy[resultPositionCopy[i][3]:resultPositionCopy[i][2], resultPositionCopy[i][1]:resultPositionCopy[i][0]]
        classLable = predictProcess.hhh(cropImage)
        print(classLable)
        cv2.rectangle(imgCopy, (resultPosition[i][1], resultPosition[i][3]), (resultPosition[i][0], resultPosition[i][2]),(0,255,0),3)
        font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
        textYaxisPosition = resultPosition[i][3] + 30
        textXaxisPosition = resultPosition[i][1] + 5
        imgzi = cv2.putText(imgCopy, classLable[0], (textXaxisPosition, textYaxisPosition), font, 1.2, (255, 255, 255), 2)
                  # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度
    cv2.imshow("Result", imgzi)
    cv2.waitKey(3)
    cv2.imwrite("segs.jpg", imgzi)











