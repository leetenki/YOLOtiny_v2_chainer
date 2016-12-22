import numpy as np
import chainer
import chainer.functions as F
import cv2

# x, y, w, hの4パラメータを保持するだけのクラス
class Box():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

# 2本の線の情報を受取り、被ってる線分の長さを返す。あくまで線分
def overlap(x1, len1, x2, len2):
    len1_half = len1/2
    len2_half = len2/2

    left = max(x1 - len1_half, x2 - len2_half)
    right = min(x1 + len1_half, x2 + len2_half)

    return right - left

# 2つのboxを受け取り、被ってる面積を返す(intersection of 2 boxes)
def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.x, a.h, b.x, b.h)
    if w < 0 or h < 0:
        return 0

    area = w * h
    return area

# 2つのboxを受け取り、合計面積を返す。(union of 2 boxes)
def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u

# compute iou
def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)

def sigmoid(x):
    return 1.0 / (np.exp(-x) + 1.0)

def softmax(x):
    x = np.array([x]) # reshape (len(x),) to (1, len(x))
    return F.softmax(x).data

def forward_cnn(model, im_org, img_width, img_height, n_grid_x, n_grid_y, n_bbox, n_classes):
    img = cv2.cvtColor(im_org, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_height, img_width)) # (416, 416, 3)
    img = np.asarray(img, dtype=np.float32) / 255.0

    ans = model.predict(img.transpose(2, 0, 1).reshape(1, 3, img_height, img_width)).data[0] # (125, 13, 13)
    ans = ans.transpose(1, 2, 0) # (13, 13, 125)
    ans = ans.reshape(n_grid_y, n_grid_x, n_bbox, (n_classes + 5)) # (13, 13, 5, 25)
    return ans

# cnnの処理結果を解釈して、box情報に変換する
def get_detected_boxes(ans, n_grid_x, n_grid_y, n_bbox, n_classes, prob_thresh, img_width, img_height, biases):
    detected_boxes = []
    grid_width = img_width / float(n_grid_x)
    grid_height = img_height / float(n_grid_y)

    for grid_y in range(n_grid_y):
        for grid_x in range(n_grid_x):
            for i in range(n_bbox):
                box = ans[grid_y, grid_x, i, 0:4] # (4,)
                conf = sigmoid(ans[grid_y, grid_x, i, 4]) 
                probs = softmax(ans[grid_y, grid_x, i, 5:])[0] # (20,)

                p_class = probs * conf # (20,)
                if np.max(p_class) < prob_thresh:
                    continue

                class_id = np.argmax(p_class)
                x = (grid_x + sigmoid(box[0])) * grid_width
                y = (grid_y + sigmoid(box[1])) * grid_height
                w = np.exp(box[2]) * biases[i][0] * grid_width
                h = np.exp(box[3]) * biases[i][1] * grid_height
                b = Box(x, y, w, h)

                detected_boxes.append([b, class_id, max(p_class)])

    return detected_boxes

def sort_boxes(boxes):
    from operator import itemgetter
    return sorted(boxes, key=itemgetter(1, 2), reverse=True) # boxes[2]とboxes[3]を使ったソート(class_idごとに、probability大きい順)

# non maximum suppression
def nms(sorted_boxes, iou_thresh):
    import itertools
    import copy

    nms_boxes = copy.copy(sorted_boxes)
    for a_sb, b_sb in itertools.combinations(sorted_boxes, 2): # 2つのboxから成る全ての組合せパターンを作る
        a = a_sb
        b = b_sb
        # 2つのclassが同じid、かつiouがthresh以上、かつ前のboxよりもprobabilityが大きい時、probの小さいほうを削除する
        if a[1] == b[1] and box_iou(a[0], b[0]) > iou_thresh and a[2] > b[2]:
            if b in nms_boxes:
                nms_boxes.remove(b)

    return nms_boxes

# clip box to object(はみ出した部分の座標を、エッジと一致させる。更にラベルを付ける。)
def clip_objects(boxes, img_width, img_height):
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
              "bus", "car", "cat", "chair", "cow",
              "diningtable", "dog", "horse", "motorbike", "person",
              "pottedplant", "sheep", "sofa", "train","tvmonitor"] 

    clipped_objects = []
    for box in boxes:
        b, class_id, p_class = box
        label = classes[class_id]
        prob = p_class * 100
        half_box_width = b.w / 2.0
        half_box_height = b.h / 2.0
        x0, y0, x1, y1 = (
            int(np.clip(b.x - half_box_width, 0, img_width)),
            int(np.clip(b.y - half_box_height, 0, img_height)),
            int(np.clip(b.x + half_box_width, 0, img_width)),
            int(np.clip(b.y + half_box_height, 0, img_height))
        )
        clipped_objects.append([(x0, y0, x1, y1), label, prob])

    return clipped_objects

def predict(model, im_org):
    img_width = 416
    img_height = 416
    n_grid_x = 13
    n_grid_y = 13
    n_classes = 20
    n_bbox = 5
    biases = [[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]]
    prob_thresh = 0.2
    iou_thresh = 0.05
    org_img_height, org_img_width = im_org.shape[0:2]

    # forward
    ans = forward_cnn(model, im_org, img_width, img_height, n_grid_x, n_grid_y, n_bbox, n_classes)

    # compute detected boxes
    detected_boxes = get_detected_boxes(ans, n_grid_x, n_grid_y, n_bbox, n_classes, prob_thresh, org_img_width, org_img_height, biases)

    # sort boxes by class_id
    sorted_boxes = sort_boxes(detected_boxes)

    # non maximum suppression
    boxes = nms(sorted_boxes, iou_thresh)

    # clip objects
    clipped_objects = clip_objects(boxes, org_img_width, org_img_height)

    return clipped_objects
