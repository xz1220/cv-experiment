import os
import cv2
import random
import numpy as np
from sklearn import svm
from sklearn.externals import joblib

if not os.path.exists("./visualizations"):
    os.makedirs("./visualizations")

data_path="../../data/"
train_path_pos=data_path+"caltech_faces/Caltech_CropFaces"
non_face_scn_path=data_path+"train_non_face_scenes"
test_scn_path=data_path+"test_scenes/test_jpg"
label_path=data_path+"test_scenes/ground_truth_bboxes.txt"

def init_hog_svm():
    
    print("init starting .....")
    winSize = (36,36)
    blockSize = (36,36) 
    blockStride = (3,3) 
    cellSize = (3,3)
    nBin = 9 

    # 创造一个HOG描述子和检测器
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nBin)

    print("hog  init finished ....")
    return hog



def get_positive_feature(file_path,hog):
    files=os.listdir(file_path)
    images=[]
    positive_features=[]
    print("starting loading the positive files and compute the hog features ....")
    for file in files:
        if not os.path.isdir(file):
            image=cv2.imread(file_path+'/'+file)
            images.append(image)
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            # hist=hog.compute(image,(3,3))
            hist=hog.compute(image)
            positive_features.append(hist)
            
    print("compute the positive hog features finished ....\n")
    return images,positive_features


def get_negative_features(file_path,hog):
    files=os.listdir(file_path)
    images=[]
    negative_features=[]
    print("starting loading the positive files and compute the hog features ....")
    for file in files:
        if not os.path.isdir(file):
            image=cv2.imread(file_path+'/'+file)
            try:
                gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            except:
                print("skip the wrong file:",file_path+'/'+file)
                continue
            h,w=gray.shape[0],gray.shape[1]
            index_x,index_y=random.randint(0,h-36),random.randint(0,w-36)
            image=image[index_x:index_x+36,index_y:index_y+36,:]
            images.append(image)
            # hist=hog.compute(image,(3,3))
            hist=hog.compute(image)
            negative_features.append(hist)
    print("compute the negative hog features finished ....\n")
    return images,negative_features

def area(box):
      return (abs(box[2] - box[0])) * (abs(box[3] - box[1]))

def overlaps(a, b, thresh=0.5):
  x1 = np.maximum(a[0], b[0])
  x2 = np.minimum(a[2], b[2])
  y1 = np.maximum(a[1], b[1])
  y2 = np.minimum(a[3], b[3])
  intersect = float(area([x1, y1, x2, y2]))
  return intersect / 512 >= thresh


def non_max_suppression_fast(boxes, overlapThresh = 0.5):
  # if there are no boxes, return an empty list
  if len(boxes) == 0:
    return []

  scores = boxes[:,4]
  score_idx = np.argsort(scores)#返回scores的从小到大排序的  索引值 
  to_delete = []
  while len(score_idx) > 0:
    box = score_idx[0]
    for s in score_idx:
      if s == score_idx[0]:
        continue
      if (overlaps(boxes[s], boxes[box], overlapThresh)):
        to_delete.append(s)
        a = np.where(score_idx == s)
        score_idx = np.delete(score_idx,a)

    score_idx = np.delete(score_idx,0)
  boxes = np.delete(boxes,to_delete,axis=0)
  return boxes

def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])#哪个维度超纲，哪个维度就显示原图

def main(i):
    if not os.path.exists("train_model.m"):
        hog =init_hog_svm()

        _,positive_features = get_positive_feature(train_path_pos,hog)
        _ , negative_features = get_negative_features(non_face_scn_path,hog)

        # 正负样本不均匀
        for _ in range(int(len(positive_features)/len(negative_features))):
            _,temp=get_negative_features(non_face_scn_path,hog)
            negative_features.extend(temp)

        print("positive_features:",positive_features[0].shape,len(positive_features))
        print("negative_features:",negative_features[0].shape,len(negative_features),"\n")

        # 生成数据以及标签
        print("starting generate the data and label ....")
        label=[]
        for _ in range(len(positive_features)):
            label.append(1)
        for _ in range(len(negative_features)):
            label.append(-1)
        for features in negative_features:
            positive_features.append(features)

        trainset=np.array(positive_features)
        trainset=trainset.reshape(trainset.shape[0],trainset.shape[1])
        print("finished generate the data && label ...... \n")


    
    # if True:
        print("train model && save it .....\n")
        clr=svm.SVC(decision_function_shape='ovo',probability=True)
        
        # print(trainset.shape,trainset[0])
        clr.fit(trainset,np.array(label))
        joblib.dump(clr, "train_model.m")
    else:
        print("load existing model .....\n")
        clr=joblib.load("train_model.m")
    
    # result=clr.predict(test_image_feats)
    # print(result.shape)
    print("starting test the model ..... ")
    count=0
    negat=0
    scale = 1
    rectangles=[]
    image_name="cs143_2011_class_easy"
    test_image=cv2.imread("../../data/extra_test_scenes/"+image_name+".jpg")
    test_image=cv2.resize(test_image,(int(test_image.shape[1]*scale),int(test_image.shape[0]*scale)))
    
    if not os.path.exists(image_name+"_windows_scale_%f.npy"%(scale)):
    # if True:
        print("creating the windows data .....\n")
        for (x, y, roi) in sliding_window(test_image, 6, (36, 36)):
            if roi.shape[1] != 36 or roi.shape[0] != 36:         #判断是否超纲
                continue
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # gray = cv2.resize(gray,(int(gray.shape[0]/2),int(gray.shape[1]/2)))
            # gray = cv2.equalizeHist(gray)

            hist=hog.compute(gray)
            test_feature=np.array(hist).reshape([1,len(hist)])
            # print(test_feature.shape)
            result=clr.predict(test_feature)
            score = clr.predict_proba(test_feature)
            # print(result,score)
            if result==1:
                count+=1
                rx, ry, rx2, ry2 = int(x * scale), int(y * scale), int((x+36) * scale), int((y+36) * scale)
                rectangles.append([rx, ry, rx2, ry2,score[0][1]])
            else:
                negat+=1

        print(count)
        print(negat)

        windows = np.array(rectangles)
        np.save(image_name+"_windows_scale_%d.npy"%(scale),windows)

    else:
        print("load existing windows file .....\n")
        windows=np.load(image_name+"_windows_scale_%f.npy"%(scale))


    windows = non_max_suppression_fast(windows,0.2)
    # print(len(boxes))
    print("finishing .....")
    for (x, y, x2, y2, score) in windows:
        cv2.rectangle(test_image, (int(x),int(y)),(int(x2), int(y2)),(0, 255, 0), 1)
        # cv2.putText(test_image, "%f" % score, (int(x),int(y)), font, 1, (0, 255, 0))
    cv2.imwrite("./test%d_scale_%f.jpg"%(i,scale),test_image)
    # print(images[0][1:10,1:10,1])

    

if __name__ == '__main__':
    print("./test%d.jpg"%(1))
    for i in range(1):
        main(i)

