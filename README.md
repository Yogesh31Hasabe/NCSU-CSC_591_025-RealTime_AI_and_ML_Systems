# CSC 591/791 (025) Fall 2024: Real-time AI and Machine Learning Systems

### Student Information
- **Name:** Yogesh Hasabe
- **Unity ID:** yhasabe
- **Email:** yhasabe@ncsu.edu

- 

## Project 0: Deep Neural Networks (DNN) and PyTorch

### What modifications does the Colab Notebook include? 

#### 1. **CNN Model Architecture Enhancements**
   - **Increased Depth of CNN:** 
     - Added 2 additional Convolutional layers, bringing the total to **4 CNN layers**.
     - This enhancement helps in capturing more complex features from the input images, leading to improved model performance.
   - **Fully Connected Layers:**
     - Included an additional Fully Connected (FC) layer, increasing the total number to **3 FC layers**.
     - This addition allows the model to better learn the representations before making the final predictions.
   - **Dropout Layers:**
     - Integrated **2 Dropout layers** to prevent overfitting and enhance generalization.
   - **Activation and Pooling Layers:**
     - Added ReLU activation functions and MaxPooling layers at appropriate stages to introduce non-linearity and reduce dimensionality.

#### 2. **Optimization Technique**
   - **Adam Optimizer:**
     - Replaced the Stochastic Gradient Descent (SGD) optimizer with the **Adam optimizer**, which is known for its adaptive learning rate and better performance on a variety of tasks.
     - Configuration:
       - Learning Rate (`lr`): **0.005**
       - Weight Decay: **5e-4**


### What is the accuracy? 

- **Achieved Accuracy:** The modifications to the model architecture and optimization technique resulted in a significant performance boost, achieving an accuracy of **86.98%** on the test dataset.


------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## Project 1: Tensor IR and Transformations

### Tabular Comparison of Execution Time
| Method            | Original | Manual Optimization | Auto Optimization |
|-------------------|----------|---------------------|-------------------|
| Convolution Method| 2.406 ms | 2.245 ms            | 0.680 ms          |
| GEMM Method       | 2.153 ms | 2.388 ms            | 0.486 ms          |

### Tabular Comparison of Accuracy
| Method            | Original | Manual Optimization | Auto Optimization |
|-------------------|----------|---------------------|-------------------|
| Convolution Method| 79.17%   | 79.17%              | 78.44%            |
| GEMM Method       | 79.17%   | 79.17%              | 78.44%            |

### Result Analysis

#### Execution Time Analysis
- The original convolution method took **2.406 ms**, and manual optimizations reduced it slightly to **2.245 ms**, offering a modest improvement. Auto optimization significantly reduced the time to **0.680 ms**, a **72%** decrease compared to the original and a **69.7%** reduction over manual optimization.
- For the GEMM method, the original execution time was **2.153 ms**, while manual optimization increased the time to **2.388 ms**, indicating suboptimal tuning. However, auto optimization led to a substantial improvement, lowering the time to **0.486 ms**, a **77.4%** reduction compared to the original and a **79.6%** reduction over manual optimization.

#### Accuracy Analysis
- The convolution method's accuracy stayed at **79.17%** for both the original and manually optimized versions, with a small decrease to **78.44%** in auto optimization, representing a **0.73%** drop.
- Similarly, the GEMM method maintained **79.17%** accuracy in both the original and manual optimizations, with auto optimization also dropping slightly to **78.44%**, matching the convolution method's **0.73%** decrease.

### Summary
- **Execution Time Improvements:** Auto optimization provided substantial performance improvements, reducing execution time by **72%** for the convolution method and by **77.4%** for the GEMM method, showing its effectiveness over manual tuning.
- **Accuracy Trade-off:** Auto optimization resulted in a small accuracy decrease of **0.73%** for both methods. This trade-off may be acceptable in scenarios where execution speed is prioritized over minor accuracy losses.

Auto optimization proves to be the most effective approach for improving execution times in both methods, despite the minor trade-off in accuracy.
