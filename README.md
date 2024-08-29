# CSC 591/791 (025) Fall 2024: Real-time AI and Machine Learning Systems

## Project 0: Deep Neural Networks (DNN) and PyTorch

### Student Information
- **Name:** Yogesh Hasabe
- **Unity ID:** yhasabe
- **Email:** yhasabe@ncsu.edu



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


