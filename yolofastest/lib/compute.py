

def compute_iou(gt_box,b_box):
    '''
    计算iou
    :param gt_box: ground truth gt_box = [x0,y0,x1,y1]（x0,y0)为左上角的坐标（x1,y1）为右下角的坐标
    :param b_box: bounding box b_box 表示形式同上
    :return: 
    '''
    width0=gt_box[2]-gt_box[0]
    height0 = gt_box[3] - gt_box[1]
    width1 = b_box[2] - b_box[0]
    height1 = b_box[3] - b_box[1]
    max_x =max(gt_box[2],b_box[2])
    min_x = min(gt_box[0],b_box[0])
    width = max(0,width0 + width1 -(max_x-min_x))
    max_y = max(gt_box[3],b_box[3])
    min_y = min(gt_box[1],b_box[1])
    height = max(0,height0 + height1 - (max_y - min_y))
    interArea = width * height
    boxAArea = width0 * height0
    boxBArea = width1 * height1
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou

def isCorrect(gt,b,thre):
    iou = compute_iou([float(gt[1]),float(gt[2]),float(gt[3]),float(gt[4])],[float(b[1]),float(b[2]),float(b[3]),float(b[4])])
    print(gt[0],b[0],iou,thre)
    if int(gt[0]) != int(b[0]):
        print("label is not equal")
        return False
    if iou < thre:
        print("thread is not equal")
        return False
    
    print("is True")
    return True

if __name__ =="__main__":
    gt = []
    predict = []
    with open("image/000000.txt","r") as f:
        while True:
            data = f.readline()
            if not data:
                break
            if data == '\n':
                break
            data = data.split(' ')
            gt.append([data[0], float(data[1]), float(data[2]), float(data[3]), float(data[4])])
    print(gt)
           
