Breast Cancer Classification
============================
  * Google Document for Initial Paper: [AI Final Project - Cancer Classifier - Andrew Kulpa](https://docs.google.com/document/d/1XtlSw0b1myOK-ZS2T91H5fVoD4ZpXZ7IUoF7B2YUeag/edit?usp=sharing)
  * Presentation: [Breast Cancer Classification: Using a Multi-layered Neural Network](https://docs.google.com/presentation/d/1J2SKmBbwqtc5LAA7hEretQtpXe4O_-S5hDtN1I1ppg0/edit?usp=sharing)
## Problem

According to [BreastCancer.org](http://www.breastcancer.org/symptoms/understand_bc/statistics), 1 in 8 United 
States women will develop invasive breast cancer in their lifetime. Early and accurate identification and 
treatment can be trivially shown to improve survival rates by the [statistical evidence](https://www.cancer.org/cancer/breast-cancer/understanding-a-breast-cancer-diagnosis/breast-cancer-survival-rates.html) 
that survival rates are much higher for treatment during earlier stages. Of course, it is also important 
to be able to recognize whether a form of breast cancer is malignant or benign. This information 
can prove vital for doctors when determining the next steps in the treatment of the tumor. For instance, 
as stated on [NationalBreastCancer.org](http://www.nationalbreastcancer.org/breast-tumors) a malignant 
tumor may require a biopsy to determine severity while a benign tumor may be left alone unless other 
complications exist.

In an artificial intelligence standpoint, this is a simple classification problem. With this in mind, the hope of this 
experiment is to demonstrate the possibility of an error rate that is lower than the typical rate of misclassification of 
breast cancer. According to a [Washington Post](https://www.washingtonpost.com/national/health-science/20-percent-of-patients-with-serious-conditions-are-first-misdiagnosed-study-says/2017/04/03/e386982a-189f-11e7-9887-1a5314b56a08_story.html?noredirect=on&utm_term=.b8b45332d5fa) 
article, the rate of cancer misdiagnosis is "more than 20 percent [for] patients who sought a second opinion". 
This is incredibly high for such a potentially life threatening diagnosis.

The idea is that by utilizing a standard set of donated data from breast cancer prognoses as input along with 
a trained doctor's actual prognoses, an artificial neural network could be trained to identify whether a 
breast tumor is malignant or benign. Of course, this is limited by the amount, availability, and recency of data. Due to 
the inherent constraints on patient data availability resulting from [HIPAA](https://www.hhs.gov/hipaa/for-individuals/guidance-materials-for-consumers/index.html) 
and other patient privacy acts, the most recent relatively large data set I could find was donated in 1995. 
I will use this data set, along with recent and common artificial intelligence methods to demonstrate the 
applicability of artificial neural networks to breast cancer classification.

## Bibliography
#### [UCI Machine Learning Repository: Wisconsin Breast Cancer Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
  * **Content:** This source includes descriptions of data, the 32 attribute dataset, and references to related work.
  * **Relevance:** The UCI Machine Learning Repository is a central hub utilized by the machine learning community for analysis of machine learning algorithms. It is the primary source for the data utilized in this project.

## Data Set 
### UCI Breast Cancer Data
  * [UCI Machine Learning Repository Link](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
  * **Date:** 
    * Donated on November 1, 1995.
  * **Entries:**
    *  569
  * **Creators:**
    * Dr. William H. Wolberg, General Surgery Dept. 
        University of Wisconsin, Clinical Sciences Center 
        Madison, WI 53792 
        wolberg@eagle.surgery.wisc.edu 
    * W. Nick Street, Computer Sciences Dept. 
      University of Wisconsin, 1210 West Dayton St., Madison, WI 53706 
      street@cs.wisc.edu 608-262-6619 
    * Olvi L. Mangasarian, Computer Sciences Dept. 
      University of Wisconsin, 1210 West Dayton St., Madison, WI 53706 
      olvi@cs.wisc.edu 
  * **Description per UCI:**
    * Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.
    * Separating plane described above was obtained using Multisurface Method-Tree (MSM-T) *[K. P. Bennett, "Decision Tree Construction Via Linear Programming." Proceedings of the 4th Midwest Artificial Intelligence and Cognitive Science Society, pp. 97-101, 1992]*, a classification method which uses linear programming to construct a decision tree. Relevant features were selected using an exhaustive search in the space of 1-4 features and 1-3 separating planes. 
    * The actual linear program used to obtain the separating plane in the 3-dimensional space is that described in: *[K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34]*. 
  * **32 Attributes**
    1. (1) ID number 
    2. (2) Diagnosis (M = malignant, B = benign) 
    3. (3-32) ten real-valued attributes for each 3 cells
        1. radius (mean of distances from center to points on the perimeter)
    	2. texture (standard deviation of gray-scale values)
    	3. perimeter
    	4. area
    	5. smoothness (local variation in radius lengths)
    	6. compactness (perimeter^2 / area - 1.0)
    	7. concavity (severity of concave portions of the contour)
    	8. concave points (number of concave portions of the contour)
    	9. symmetry 
    	10. fractal dimension ("coastline approximation" - 1)
    
  

## Methodology
From the initial 569 records, 499 records (305 Benign, 194 Malignant) were used for training. The remaining 70 records (52 Benign, 18 Malignant) were used for testing. Network processing included a one-hot encoding of these labels. In addition to this, the network also reformatted the 30 attributes resulting from the 3 cell samples using an appended padding row to reshape them into 6x6 matrices. This was done in order to ease the implementation of image processing methods on this data. In the current implementation and further iterations thus far, the last 6 appended zero inputs were kept for consistency reasons for possible future integrations of input data into a convolutional neural network.
The neural network functions as a multi-layered perceptron. The system is comprised of an input layer of 36 nodes, two hidden layers of 100 nodes each, and an output layer of 2 nodes. There are weights and biases between each layer as this is a fully connected neural network. The 36 input nodes are fed the 30 attributes from the 3 cell samples per tumor along with the aforementioned 6 zero inputs as a padding for proposed future classification using image processing techniques via a convolutional neural network approach. The system will output two nodes which will be used for softmax classification. The softmax output is based upon the two classes of cancer presented here, including Benign and Malignant. This predicted output is then compared against one-hot encoded training target data.
All weights and biases are initially created using automatically generated a Tensorflow random normal number generation method. An Adam optimization algorithm was utilized instead of a Gradient Descent algorithm. The general consensus among some of the machine learning blogs and forums is that the Adam optimization algorithm is quick and effective. As such, its usage was in part a result of community recommendation. A drawing of the model is presented below:

### System Flow
The neural network functions as a multi-layered perceptron. The system is comprised of
an input layer of 36 nodes, two hidden layers of 100 nodes each, and an output layer 
of 2 nodes. 
#### Input / Output
The 36 input nodes are fed the 30 attributes from the 3 cell samples per tumor along 
with the aforementioned 6 zero inputs as a padding for proposed future classification 
using image processing techniques via a convolutional neural network approach. The system 
will output two nodes which will be used for softmax classification. The softmax output 
is based upon the two classes of cancer presented here, including Benign and Malignant. 
This predicted output is then compared against one-hot encoded training target data.
### Implementation
This initial implementation utilized a learning rate of 0.03 and was trained in 100 epoch. 
All weights and biases are initially created using automatically generated a Tensorflow 
random normal number generation method. An Adam optimization algorithm was utilized 
instead of a Gradient Descent algorithm. This was done by recommendation of multiple 
machine learning forums and blogs, such as [machinelearningmastery.com](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/). 
In practice, it is incredibly effective.

