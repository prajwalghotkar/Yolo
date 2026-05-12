import os, uuid
import numpy as np
from numpy import expand_dims
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import cv2

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/results', exist_ok=True)

MODEL = None

LABELS = [
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana",
    "apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
    "chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

ANCHORS = [
    [116,90, 156,198, 373,326],
    [30,61, 62,45, 59,119],
    [10,13, 16,30, 33,23]
]

COLORS = [
    (0,255,136),(255,51,102),(255,204,0),(0,204,255),(255,153,0),
    (204,0,255),(0,255,68),(255,0,153),(0,255,204),(255,102,0),
    (0,153,255),(255,255,0),(255,0,68),(102,255,0),(255,51,255)
]

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin=xmin; self.ymin=ymin; self.xmax=xmax; self.ymax=ymax
        self.objness=objness; self.classes=classes
        self.label=-1; self.score=-1
    def get_label(self):
        if self.label==-1: self.label=np.argmax(self.classes)
        return self.label
    def get_score(self):
        if self.score==-1: self.score=self.classes[self.get_label()]
        return self.score

def _sigmoid(x): return 1./(1.+np.exp(-x))

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    boxes = []
    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4:] = _sigmoid(netout[..., 4:])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh
    for i in range(grid_h * grid_w):
        row = i / grid_w; col = i % grid_w
        for b in range(nb_box):
            objectness = netout[int(row)][int(col)][b][4]
            if objectness.all() <= obj_thresh: continue
            x, y, w, h = netout[int(row)][int(col)][b][:4]
            x = (col + x) / grid_w
            y = (row + y) / grid_h
            w = anchors[2*b+0] * np.exp(w) / net_w
            h = anchors[2*b+1] * np.exp(h) / net_h
            classes = netout[int(row)][int(col)][b][5:]
            boxes.append(BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes))
    return boxes

def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    for b in boxes:
        b.xmin = int(b.xmin * image_w)
        b.xmax = int(b.xmax * image_w)
        b.ymin = int(b.ymin * image_h)
        b.ymax = int(b.ymax * image_h)

def _interval_overlap(a, b):
    x1,x2=a; x3,x4=b
    if x3<x1: return 0 if x4<x1 else min(x2,x4)-x1
    return 0 if x2<x3 else min(x2,x4)-x3

def bbox_iou(b1, b2):
    iw=_interval_overlap([b1.xmin,b1.xmax],[b2.xmin,b2.xmax])
    ih=_interval_overlap([b1.ymin,b1.ymax],[b2.ymin,b2.ymax])
    inter=iw*ih
    u=(b1.xmax-b1.xmin)*(b1.ymax-b1.ymin)+(b2.xmax-b2.xmin)*(b2.ymax-b2.ymin)-inter
    return float(inter)/u if u>0 else 0

def do_nms(boxes, thresh):
    if not boxes: return
    for c in range(len(boxes[0].classes)):
        si=np.argsort([-b.classes[c] for b in boxes])
        for i in range(len(si)):
            if boxes[si[i]].classes[c]==0: continue
            for j in range(i+1,len(si)):
                if bbox_iou(boxes[si[i]],boxes[si[j]])>=thresh:
                    boxes[si[j]].classes[c]=0

def get_boxes(boxes, labels, thresh):
    vb,vl,vs=[],[],[]
    for box in boxes:
        for i,label in enumerate(labels):
            if box.classes[i]>thresh:
                vb.append(box); vl.append(label); vs.append(box.classes[i]*100)
    return vb,vl,vs

def load_image_pixels(path, shape):
    img=load_img(path); w,h=img.size
    img=load_img(path,target_size=shape)
    arr=img_to_array(img).astype('float32')/255.0
    return expand_dims(arr,0), w, h

def draw_result(in_path, v_boxes, v_labels, v_scores, out_path):
    img=cv2.imread(in_path)
    lmap={}; ci=0
    for i in range(len(v_boxes)):
        box=v_boxes[i]; label=v_labels[i]; score=v_scores[i]
        if label not in lmap: lmap[label]=COLORS[ci%len(COLORS)]; ci+=1
        c=lmap[label]
        x1,y1=max(0,box.xmin),max(0,box.ymin)
        x2,y2=min(img.shape[1],box.xmax),min(img.shape[0],box.ymax)
        cv2.rectangle(img,(x1,y1),(x2,y2),c,3)
        txt=f"{label} {score:.1f}%"
        (tw,th),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.65,2)
        cv2.rectangle(img,(x1,y1-th-14),(x1+tw+10,y1),c,-1)
        cv2.putText(img,txt,(x1+5,y1-7),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,0),2)
    cv2.imwrite(out_path,img)

def get_model():
    global MODEL
    if MODEL is None:
        MODEL=load_model('model.h5')
    return MODEL

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error':'No image uploaded'}),400
    f=request.files['image']
    if not f or f.filename=='':
        return jsonify({'error':'No file selected'}),400

    ext=os.path.splitext(f.filename)[1].lower()
    uid=str(uuid.uuid4())
    in_path=f'static/uploads/{uid}{ext}'
    out_path=f'static/results/result_{uid}{ext}'
    f.save(in_path)

    try:
        model=get_model()
        iw,ih=416,416
        image,image_w,image_h=load_image_pixels(in_path,(iw,ih))
        yhat=model.predict(image)

        thresh=float(request.form.get('threshold',0.6))
        boxes=[]
        for i in range(len(yhat)):
            boxes+=decode_netout(yhat[i][0],ANCHORS[i],thresh,ih,iw)
        correct_yolo_boxes(boxes,image_h,image_w,ih,iw)
        do_nms(boxes,0.5)

        vb,vl,vs=get_boxes(boxes,LABELS,thresh)
        draw_result(in_path,vb,vl,vs,out_path)

        dets=[{
            'label':vl[i],
            'score':round(float(vs[i]),2),
            'box':{'xmin':int(vb[i].xmin),'ymin':int(vb[i].ymin),'xmax':int(vb[i].xmax),'ymax':int(vb[i].ymax)}
        } for i in range(len(vb))]

        return jsonify({
            'success':True,
            'result_image':'/'+out_path,
            'original_image':'/'+in_path,
            'detections':dets,
            'total':len(dets)
        })
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/test')
def test_zebra():
    return jsonify({'status':'ok','message':'Use POST /detect with an image'})

if __name__=='__main__':
    port=int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0',port=port,debug=False)
