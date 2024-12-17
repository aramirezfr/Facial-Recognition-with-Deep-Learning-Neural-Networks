# Business Understanding 
The goal of this project is to develop a binary image classifier capable of accurately distinguishing between human faces and non-face objects, addressing a critical need in security and surveillance systems. By leveraging advanced machine learning techniques, the project aims to create a reliable model that can be seamlessly integrated into real-world applications, such as smart surveillance systems, biometric authentication, and automated monitoring. Accurate differentiation between faces and objects is essential for ensuring public safety, operational efficiency, and privacy compliance. Failure to effectively distinguish these classes can result in security risks, inefficiencies, and unnecessary resource allocation. By tackling these challenges, the project has the potential to enhance the reliability of surveillance systems, minimize errors, and contribute to the broader adoption of AI-driven technologies in security and access control.

# Data Understanding
This project utilizes two distinct datasets: the Tiny ImageNet dataset and the Labeled Faces in the Wild (LFW) dataset. The Tiny ImageNet dataset contains 200 object classes with over 100,000 RGB training images and 10,000 validation images, all resized to 64x64 pixels. Its diversity in object types makes it highly suitable for representing non-face entities, which is essential for training a robust binary classifier. The LFW dataset, on the other hand, comprises more than 13,000 face images collected under various lighting conditions, orientations, and resolutions. Designed for face recognition tasks, it provides an extensive set of labeled face images, enabling the model to learn distinctive facial features.  

Combining these datasets ensures balanced representation between face and object classes, a critical factor for reducing bias and improving model performance. Features such as RGB pixel intensities capture the raw visual data necessary for distinguishing the two classes, while the variation in lighting, orientation, and object types enhances the model’s generalizability to real-world scenarios. Both datasets are well-labeled, facilitating clear distinctions for binary classification tasks. With appropriate preprocessing steps, including normalization and balancing of classes, these datasets provide a robust foundation for developing a high-performing classifier. For additional details on the datasets, refer to their Kaggle pages: [LFW Dataset](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset) and [Tiny ImageNet Dataset](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet/data).

## Data Preparation
For this project, several data preparation techniques were employed to ensure the datasets were ready for model training and evaluation. **Data normalization** was applied to scale RGB pixel intensities to a range of [0, 1], ensuring consistent input for the model and improving convergence during training. **Data augmentation** techniques such as random cropping, horizontal flipping, and rotation were used to artificially increase the size of the dataset and introduce variability, helping the model generalize better to unseen data. To address potential class imbalances, **class balancing** was implemented by oversampling the smaller class or undersampling the larger class as needed. Additionally, **data splitting** was performed to divide the datasets into training, validation, and test sets, ensuring the model was evaluated on data it had not encountered during training. These preparation steps collectively enhanced the quality of the input data, enabling the development of a more robust and accurate classifier.

# Exploratory data analysis 
The modeling section of this project explored three distinct models to classify images into "faces" or "objects." Each model's performance and characteristics were analyzed to understand its behavior and effectiveness for this task.  

#### **1. Simple Decision Tree and Random Forest Models**  
- **Decision Tree**:  
  A simple decision tree was implemented as an initial model to establish a basic understanding of the classification task.  
  - **Key Insights**:  
    - The decision tree performed poorly due to its inability to handle the high-dimensional and complex pixel data effectively.  
    - Overfitting was a major issue, as the model memorized training data but failed to generalize to new images.  

- **Random Forest**:  
  A simple random forest model was introduced to improve upon the decision tree by aggregating multiple weak learners.  
  - **Key Insights**:  
    - The random forest model showed better performance than the decision tree, leveraging ensemble learning to reduce variance and improve accuracy.  
    - However, like the decision tree, it struggled with the complexity of image data, indicating a need for feature extraction methods or dimensionality reduction.  

#### **2. Baseline Model (Convolutional Neural Network)**  
A baseline CNN model was developed to leverage deep learning techniques for feature extraction and classification.  
- **Key Insights**:  
  - The CNN significantly outperformed the decision tree and random forest models, capturing spatial and hierarchical features from the image data.  
  - Overfitting was observed, especially during the later epochs, suggesting a need for regularization methods such as dropout.  
  - This model set a benchmark for the project, demonstrating the importance of deep learning in handling image classification tasks.  

#### **3. Hyperparameter-Tuned CNN Model**  
The third model extended the baseline CNN by incorporating hyperparameter tuning to optimize its performance. Techniques such as grid search and random search were used to refine parameters like the learning rate, number of filters, kernel sizes, and dropout rates.  
- **Key Insights**:  
  - The hyperparameter-tuned CNN achieved the best performance among all models, with higher accuracy, precision, and recall compared to the baseline CNN.  
  - Regularization methods like dropout effectively mitigated overfitting, enabling the model to generalize better on unseen data.  
  - Adjustments to the learning rate improved convergence, and increased filter sizes helped capture more complex features in the images.
    
# Evaluation

# Conclusionn
## Limitations
## Next Steps
## More Information:
Find the full analysis in the [GoogleColab] or review this [presentation]().

## Repository Structure
- Graphs
- README.md
- .ipynb
- .pdf

### Author: Adriana Ramirez Franco (aramirezfr20@gmail.com)
