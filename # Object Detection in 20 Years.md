# Object Detection in 20 Years

## abstract
+ object detection: where is the object

+ **two significant metrics in object detection:**

  - speed 

  - accuracy
+ **foundation in CV**
  - instance segmentation 
  - image captioning
  - image tracking
+ **necessary components**
  - datasets
  - metrics
  - acceleration techniques

## history

+ traditional methods before 2014, <mark> deep-learning </mark> based after 2014
+ **VJ detectors**
  - human face detection
  - run on 700Mhz CPU
  - Three techniques: "integral image", "feature selection", "detection cascades"
+ **HOG detector**
  - balance feature invariance (translation, scale, illumination) and nonlinearity 
  - compute on dense grid of uniformly spaced cells and use overlapping local contrast normalization
  - detection window unchanged, rescale input image

+ **Deformable Part-based model** 
  - divide and conquer
  - training: learn to decompose an object
  - inference: ensemble of detecting different object parts
  
### CNN based two_stage detectors

+ handcrafted features saturated
+ deep convulutional network is able to learn robust and high level feature representations
+ two groups of detectors
  - one stage: complete in one step
  - two stages: coarse-to-fine

+ **RCNN**
  - extraction of object proposals
  - fed into CNN model pretrained on imageNet
  - finally use SVM classifier to predict the presence of an object within each region
  - **too slow**

+ **SPPNet**
  - spp layer to generate a fixed-length representation regardless the size/region of interest without rescaling
  - no need to regain the features of one graph for training
  - one train for the whole graph and select regions of it to have fixed-length result for future tuning
  - **only fine-tuning full layer while ignore all the former layers**

+ **Fast RCNN**
  - train detector and bounding box regressor the same time
  - **restrained by proposal detection**

+ **Faster RCNN**
  - introduction of region proposal network enables cost-free region proposals

+ **Feature pyramid network**
  - features in deeper layer is beneficial for category recognition, not conducive to localizing objects
  - top-down structure with lateral connection: building semantics at all scales

### CNN based one_stage detectors

+ two stage detectors are more precise but time-consuming, not common in engineering practice. In contrast, one stage detectors are more friendly on mobile devices.


+ **YOLO**
  - apply single neutral network to the full image
  - divide the image into regions and predict each region simultaneously

+ **SSD**
  - good for small objects
  - detect objects of different scales on different layers

+ **RetinaNet**
  - foreground - background imbalance
  - new loss function "focal loss": put more focus on hard, misclassified examples

+ **CornerNet**
  - task as keypoint (corners of a box)
  - use keypoints to generate bounding boxes

+ **CenterNet**
  
  - fully end-to-end
  - consider object to be a single points (its center)
  - regress all attributes (size, orientation, location, pose) based on the reference center point
  
+ **DETR**
  
  - Transformer-based
  
### Object detection datasets and metrics

+ building larger datasets with less bias is essential for developing advanced detection algorithms

+ **Pascal VOC**
  - two versions: 2005 and 2012
  - 2005: 5k traning images and 12k annotated objects
  - 2012: 11k training images and 27k annotated objects
  - 20 classes of common objects like person, cat, ...

+ **ILSVRC**
  - detection challenge using ImageNet
  - two orders of magnitude larger than VOC
  - 200 classes and more

+ **MS-COCO**
  - more object instances: 164k images and 897k annotated objects frin 80 categories
  - more small objects whose area is smaller than 1% of the image

+ **Open Images**
  - more object instances: 1910k images and 15440k annotated objects frin 80 categories
  - two tasks: 1) standard object detection. 2) visual relation detection

### How to evaluate accuracy of a detector

+ false positives per window (FPPW) to false positives per image (FPPI).
  
+ mAP averaged over all categories is usually used as the metric of performance.

+ loU of predicted box and ground truth is used to verify whether its is greater than a predefined threhold. 0.5-loU mAP has become de facto metric of object detection.

+ After 2014 MS-COCO, shrehold becomes multiple between 0.5 and 0.95. AP is taken average of those outcomes.

## Technical evolution in object detection

### multi-scale detection

+ **Feature pyramids and sliding window. **
  - slide a detecting window on the graph, ignoring difference of aspect ratios.
  - "mixture model" train multiple detectors for objects of different aspect ratios

+ **Detection with object proposal. **
  - a group of class-agnostic reference boxes that are likely to contain any objects
  - out-of-date now

+ **Deep regression and anchor-free detection **
  - keypoint detection: 1) group-based 2) none group-based

+ **Multi-reference/-resolution detection. **
  - multi-reference: define a set of references at every locations of the image
  - multi-resolution: detecting objects of different scales at different layers of the network

### context priming

+ visual objects are embedded in a typical context with the surrounding environments

+ **Local context. **
  - refer to the visual information in the area that surrounds the object to detect
  - enlarge the receptive field or the size of object proposals to improve with local context in the deep learning based detector

+ **Global context. **
  - use deep convolution, dilated convolution deformable convolution and pooling to receive larger receptive field. now with transformer
  - think of a sequential information and learn it with RNN

+ **Context interactive **
  - constraints and dependencies that conveys between visual elements.

### Hard negative mining

+ easy negatives may overwhelm the learning process. Hard negative mining aims to overcome the problem,

+ **Bootstrap. **
  - starts with a small part of background samples and then iteratively adds new miss-classified samples
  - reducing the training computations over millions of backgrounds

+ **HNM in deep learning based detectors. **
  - bootstrap was shortly discarded in object detection during 2014-2016
  - balance the weights between the positive and negative windows in Faster RCNN and YOLO. (not enough)
  - design new loss function

### Loss function

+ measures how the model match the data. the loss yields the gradients of the model weights

$$
\mathcal{L}(p, p^*, t ,  t^*) = \mathcal{L}_{\text{cls}}(p, p^*) + \beta \cdot \mathbb{I}(t) \cdot \mathcal{L}_{\text{loc}}(t, t^*)
$$


$$
I(t) =
\begin{cases} 
1 & \text{IoU}\{a, a^*\} > \eta \\
0 & \text{else}
\end{cases}
$$

+ where t and t∗ are the locations of predicted and ground truth bounding boxes, p and p∗ are their category probabilities. IoU{a,a∗} is the IoU between the reference box/point a and its ground-truth a∗. η is an IoU threshold, say, 0.5. If an anchor box/point does not match any objects, its localization loss does not count in the final loss.

+ **Classification loss **
  - CE loss: measure distribution differences
  - For improving categorization efficiency, Label Smooth has been proposed to enhance the model gen eralization ability and solve the overconfidence problem on noise labels
  - Focal loss is designed to solve the problem of category imbalance and differences in classification difficulty

+ **Localization loss **
  - optimize position and size
  - combine L1 and L2 loss for smooth L1 as follows

$$
\text{Smooth}_{L1}(x) = 
\begin{cases} 
0.5x^2 & \text{if } |x| < 1 \\ 
|x| - 0.5 & \text{else} 
\end{cases}
\tag{2}
$$

where x denotes the difference between the target and predicted values. When calculating the error, the above losses treat four numbers (x,y,w,h) representing a bounding box as independent variables, however, a correlation exists between them. Moreover, IoU is utilized to determine if the prediction box corresponds to the actual ground truth box in evaluation. Equal Smooth L1 values will have totally different IoU values, hence IoU loss [105] is introduced as follows:

$$
\text{IoU loss} = -\log(\text{IoU})
$$

+ GIoU improved the case when IoU
loss could not optimize the non-overlapping bounding boxes i.e., IoU = 0.
+  a successful detection regression loss should meet three geometric metrics:
overlap area, center point distance, and aspect ratio

So, based on IoU loss and G-IoU loss, DIoU (Distance IoU) is defined as the distance between the center point of the prediction and the ground truth, and CIoU (Complete IoU)  considered the aspect ratio difference on the basis of DIoU.

### Non maximum suppression

As the neighboring windows usually have similar detection scores, the non-maximum suppression is used as a post-processing step to remove the replicated bounding boxes and obtain the final detection result.

+ **Greedy selection **
  - for a set of overlapped detections, the bounding box with the maximum detection score is se lected while its neighboring boxes are removed
  - the top-scoring box may not be the best fit.
  - it may suppress nearby objects
  - it does not suppress false positives

+ **Bounding Box Aggregation **
  -  combining or clustering multiple overlapped bounding boxes into one final detection.
  - takes full consideration of object relationships and their spatial layout 

+ **learning based NMS **
  - think of NMS as a filter to re-score all raw detections and to train the NMS as part of a network in an end-to-end fashion or train a net to imitate NMS’s behavior

+ **NMS-free detector **
  - complete one-to-one label assignment (a.k.a. one object with just one prediction box)

## Speed Up of Detection

“detection pipeline”, “detector backbone”, and “numerical computation”.

+ **Feature Map shared Computation **
  - computed the feature map of the hwole image only once

+ **Cascaded Detection  **
  - coarse to fine detection philosophy: to filter out most of the simple background windows using simple calculations then proceed to more difficult ones
  - applied to small objects in big scenes

+ **Network Pruning and Quantification  **
  - The former refers to pruning the network structure or weights: recently to remove only a small group of unimportant weights after each stage of training, and to repeat those operations
  - the latter refers to reducing their code length: recently focus on network binarization, which aims to compress a network by quantifying its activations or weights to binary variables (say, 0/1) so that the floating-point operation is converted to logical operations.

+ **Lightweight Network Design  **
  - Factorizing Convolutions:
     * factorize a large convolution filter into a set of small ones
     * factorize convolutions in their channel dimensions 
  - divide the feature channels into different groups. m groups indicate the computation 1/m of the original one.

+ **Depth-wise Separable Convolution **
  - a special case of the group convolution when the number of groups is set equal to the number of channels
  - 1x1 filters are used to make a dimension transform so that the final output will have the desired number of channels
  - applied to object detection and f ine-grain classification

+ **Bottle-neck Design**
  - Deep-learning rely heavily on hand crafted network architecture and training parameters
  - primarily concerned with defining the proper space of candidate networks, improving strategies for searching quickly and accurately, and validating the search ing results at a low cost.
  - reduce human work

### numerical accelerations

+ **Speed Up with Integral Image**
  - rapidly calculate summations over image sub-regions. 

$$
f(x)*g(x) = ( \int f(x)dx)) *\left( \frac{dg(x)}{dx} \right)
$$

where if dg(x)/dx is a sparse signal, then the convolution can be accelerated by the right part of this equation.

The integral image can also be used to speed up more general features in object detection

+ **Speed Up in Frequency Domain**
  - linear detector as window-wise inner product between the feature map and de tector’s weights
  - The Fourier transform is a very practical way to speed up convolutions

$$
f(x)*g(x) = ( \int f(x)dx)) *\left( \frac{dg(x)}{dx} \right)
$$

where if dg(x)/dx is a sparse signal, then the convolution can be accelerated by the right part of this equation.

The integral image can also be used to speed up more general features in object detection

$$
I \times W = F(F(I) \circ F(W))
$$

where F is Fourier transform, F^−1 is Inverse Fourier transform, and circle is the point-wise product. The above calculation can be accelerated by using the Fast Fourier Transform (FFT) and the Inverse FFT (IFFT) 

+ **Vector Quantization**
  - approximate the distribution of a large group of data by a small set of prototype vectors. It can be used for data compression and accelerating the inner product operation in object detection

### Recent Advances In Object Detection

+ **Beyond sliding window detection**
  - an object in an image can be uniquely determined by its upper left corner and lower right corner of the ground truth box: pair-wise key points localization problem
  - Another paradigm views an object as a point/points and directly predicts the object’s attributes and there is no need to design multi scale anchor boxes.

+ **Rotation Robust Detection**
  - data augmentation so that an object in any orientation can be well covered 
  - train independent detectors for each orientation
  - use polar coordinates so that features can be robust to the rotation changes

+ **Scale Robust Detection (training)**
  - Modern detectors usually use re-sclae input images to a fixed size and back propagate the loss in all scales which cause scale imbalance.
  - Building pyramid may ease but not solve the problem.
  - scale normalization for image pyramids SNIP with efficient resampling

+ **Scale Robust Detection (detection)**
  - adpative zoom in the small objects
  - Building pyramid may ease but not solve the problem.
  - predict the scale distribution of objects in an image, and then adaptively re-scaling the image according to it

+ **Detection with better backbone**
  - feature extraction network 
  - most high-performance base on transformer

+ **improvement of localization **
  - Bounding box refinement
  - new loss function

+ **learning with segmentation loss **
  - think of the segmentation network as a fixed feature extractor and to integrate it into a detector as auxiliary features
  - introduce an additional segmentation branch on top of the original detector and to train this model with multi-task loss functions

+ **Generative adversarial networks **
  - GAN can be used to enhance the features of small objects by narrowing the representations between small and large ones
  - generate occlusion masks by using adversarial training.

+ **Weakly Supervised Object Detection**
  - (WSOD) aims at easing the reliance on data annotation by training a detector with only image-level annotations instead of bounding boxes
  - multi-instance learning
  - class activation mapping
  - GAN

+ **Detection with domain adaptation**
  - offers the possibility of narrowing the gap between domains.

## Conclusion

+ Lightweight object detection
+ End-to-End object detection
+ Small object detection
+ 3D object detection
+ Detection in videos
+ Cross-modality detection

+ Towards open-world detection


