# CSC 591/791 (025) Fall 2024: Real-time AI and Machine Learning Systems

### Student Information
- **Name:** Yogesh Hasabe
- **Unity ID:** yhasabe
- **Email:** yhasabe@ncsu.edu



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




------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




## Project 2: DNN Pruning

## Overview
In this project, we explore two types of pruning techniques applied to Deep Neural Networks (DNNs): Unstructured Fine-Grained Pruning and Structured Filter Pruning. The focus is on how pruning affects model size and accuracy, both before and after fine-tuning.

### Pruning Ratios for Unstructured Fine-Grained Pruning (Before Fine-Tuning)
| Pruning Ratio | Accuracy (%) | Model Size (MiB) | Size Reduction Factor |
|---------------|--------------|------------------|-----------------------|
| 50%           | 98.58        | 2.29             | 2.00X                 |
| 60%           | 97.50        | 1.83             | 2.50X                 |
| 70%           | 95.52        | 1.37             | 3.33X                 |
| 80%           | 74.44        | 0.92             | 5.00X                 |
| 90%           | 18.01        | 0.46             | 9.98X                 |
| 95%           | 14.56        | 0.23             | 19.93X                |
| 99%           | 10.32        | 0.05             | 98.11X                |

#### Result Analysis for Unstructured Pruning:
- As pruning increases, model size reduces significantly, but accuracy drops rapidly beyond 80% sparsity.
- At 80% pruning, accuracy falls to 74.44%, with a 5.00X model size reduction.
- The highest pruning ratio of 99% leads to 10.32% accuracy, with a drastic size reduction (98.11X).

### Pruning Ratios for Unstructured Fine-Grained Pruning (After Fine-Tuning)
| Pruning Ratio | Accuracy (%) | Model Size (MiB) | Size Reduction Factor |
|---------------|--------------|------------------|-----------------------|
| 50%           | 98.75        | 2.29             | 2.00X                 |
| 60%           | 98.92        | 1.83             | 2.50X                 |
| 70%           | 98.83        | 1.37             | 3.33X                 |
| 80%           | 98.77        | 0.92             | 5.00X                 |
| 90%           | 98.13        | 0.46             | 9.98X                 |
| 95%           | 94.72        | 0.23             | 19.93X                |
| 99%           | 37.47        | 0.05             | 98.11X                |

### Fine-Tuning Results for Unstructured Fine-Grained Pruning

- **50% Sparsity**:  
  Epoch 1 Sparse Accuracy: 98.75% / Best Sparse Accuracy: 98.75%  
  Epoch 2 Sparse Accuracy: 99.14% / Best Sparse Accuracy: 99.14%

- **60% Sparsity**:  
  Epoch 1 Sparse Accuracy: 98.92% / Best Sparse Accuracy: 98.92%  
  Epoch 2 Sparse Accuracy: 99.03% / Best Sparse Accuracy: 99.03%

- **70% Sparsity**:  
  Epoch 1 Sparse Accuracy: 98.83% / Best Sparse Accuracy: 98.83%  
  Epoch 2 Sparse Accuracy: 99.01% / Best Sparse Accuracy: 99.01%

- **80% Sparsity**:  
  Epoch 1 Sparse Accuracy: 98.77% / Best Sparse Accuracy: 98.77%  
  Epoch 2 Sparse Accuracy: 99.01% / Best Sparse Accuracy: 99.01%

- **90% Sparsity**:  
  Epoch 1 Sparse Accuracy: 98.13% / Best Sparse Accuracy: 98.13%  
  Epoch 2 Sparse Accuracy: 98.48% / Best Sparse Accuracy: 98.48%

- **95% Sparsity**:  
  Epoch 1 Sparse Accuracy: 94.72% / Best Sparse Accuracy: 94.72%  
  Epoch 2 Sparse Accuracy: 96.28% / Best Sparse Accuracy: 96.28%

- **99% Sparsity**:  
  Epoch 1 Sparse Accuracy: 37.47% / Best Sparse Accuracy: 37.47%  
  Epoch 2 Sparse Accuracy: 46.22% / Best Sparse Accuracy: 46.22%

### Pruning Ratios for Structured Filter Pruning (Before Fine-Tuning)
| Pruning Ratio | Accuracy (%) | Model Size (MiB) | Size Reduction Factor |
|---------------|--------------|------------------|-----------------------|
| 50%           | 98.06        | 4.54             | 1.01X                 |
| 60%           | 96.17        | 4.53             | 1.01X                 |
| 70%           | 91.12        | 4.53             | 1.01X                 |
| 80%           | 48.92        | 4.52             | 1.01X                 |
| 90%           | 24.61        | 4.51             | 1.01X                 |
| 95%           | 18.80        | 4.51             | 1.02X                 |
| 99%           | 10.28        | 4.51             | 1.02X                 |

#### Result Analysis for Structured Filter Pruning:
- Unlike unstructured pruning, filter pruning minimally reduces model size.
- Accuracy degradation starts slowly but drops sharply after higher sparsity levels.
- At 80% pruning, accuracy drops to 48.92%, while model size remains at 1.01X of the original.

### Pruning Ratios for Structured Filter Pruning (After Fine-Tuning)
| Pruning Ratio | Accuracy (%) | Model Size (MiB) | Size Reduction Factor |
|---------------|--------------|------------------|-----------------------|
| 50%           | 98.84        | 4.54             | 1.01X                 |
| 60%           | 98.94        | 4.53             | 1.01X                 |
| 70%           | 98.84        | 4.53             | 1.01X                 |
| 80%           | 98.73        | 4.52             | 1.01X                 |
| 90%           | 98.22        | 4.51             | 1.01X                 |
| 95%           | 96.28        | 4.51             | 1.02X                 |
| 99%           | 46.22        | 4.51             | 1.02X                 |

### Fine-Tuning Results for Structured Filter Pruning

- **50% Sparsity**:  
  Epoch 1 Sparse Accuracy: 98.33% / Best Sparse Accuracy: 98.33%  
  Epoch 2 Sparse Accuracy: 98.84% / Best Sparse Accuracy: 98.84%

- **60% Sparsity**:  
  Epoch 1 Sparse Accuracy: 98.47% / Best Sparse Accuracy: 98.47%  
  Epoch 2 Sparse Accuracy: 98.94% / Best Sparse Accuracy: 98.94%

- **70% Sparsity**:  
  Epoch 1 Sparse Accuracy: 98.37% / Best Sparse Accuracy: 98.37%  
  Epoch 2 Sparse Accuracy: 98.84% / Best Sparse Accuracy: 98.84%

- **80% Sparsity**:  
  Epoch 1 Sparse Accuracy: 97.62% / Best Sparse Accuracy: 97.62%  
  Epoch 2 Sparse Accuracy: 98.73% / Best Sparse Accuracy: 98.73%

- **90% Sparsity**:  
  Epoch 1 Sparse Accuracy: 97.60% / Best Sparse Accuracy: 97.60%  
  Epoch 2 Sparse Accuracy: 98.22% / Best Sparse Accuracy: 98.22%

- **95% Sparsity**:  
  Epoch 1 Sparse Accuracy: 94.72% / Best Sparse Accuracy: 94.72%  
  Epoch 2 Sparse Accuracy: 96.28% / Best Sparse Accuracy: 96.28%

- **99% Sparsity**:  
  Epoch 1 Sparse Accuracy: 37.47% / Best Sparse Accuracy: 37.47%  
  Epoch 2 Sparse Accuracy: 46.22% / Best Sparse Accuracy: 46.22%

### Key Insights for Filter Pruning
- **Filter Dependencies**: Pruning filters alone doesn’t reduce entire connections across layers.
- **Downstream Impact**: Full layers and their connectivity remain, reducing the size impact.
- **Framework Overhead**: PyTorch or other frameworks still allocate memory for pruned elements.

### Summary
- **Unstructured Pruning**: Significant size reductions, but accuracy falls sharply after 80% sparsity.
- **Structured Pruning**: Minimal size reductions, slower accuracy degradation initially, with sharp drops at high sparsity levels.
- **Fine-Tuning**: Greatly improves accuracy in both unstructured and structured pruning, especially for higher pruning ratios.




------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

