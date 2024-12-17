# Business Understanding:
This project aims to develop a binary image classifier that can accurately distinguish between human faces and non-face objects. By using advanced machine learning techniques, the goal is to create a reliable model for applications like smart surveillance, biometric authentication, and automated monitoring. Accurate classification is crucial for public safety, operational efficiency, and privacy compliance. The project seeks to enhance surveillance system reliability, reduce errors, and support the broader adoption of AI in security and access control.

# Data Understanding:
This project uses two datasets: Tiny ImageNet and Labeled Faces in the Wild (LFW). Tiny ImageNet includes over 100,000 training images and 10,000 validation images across 200 object classes, resized to 64x64 pixels, making it ideal for non-face entities. LFW contains over 13,000 face images under various conditions, perfect for learning facial features.

Combining these datasets ensures balanced representation of faces and objects, reducing bias and improving model performance. The diversity in lighting, orientation, and object types enhances the model's generalizability. Proper preprocessing, like normalization and class balancing, provides a strong foundation for a high-performing classifier. For more details, check their Kaggle pages: LFW Dataset and Tiny ImageNet Dataset.
![Face Images vs. Non-Face Images](/Graphs/sample_images.jpg)

## Data Preparation:
For this project, data preparation involved several techniques to ensure readiness for model training and evaluation. **Class balancing** addressed imbalances by oversampling smaller classes or undersampling larger ones. **Data splitting** divided datasets into training, validation, and test sets for proper model evaluation. These steps collectively enhanced input data quality, leading to a more robust and accurate classifier.
![Data balanced](/Graphs/dataset_sizes.jpg)

# Exploratory data analysis :
The modeling section of this project explored three distinct models to classify images into "faces" or "objects." Each model's performance and characteristics were analyzed to understand its behavior and effectiveness for this task.  
- The project began with a **Decision Tree** model (Model 1), which performed poorly due to overfitting and difficulty handling complex pixel data. A **Random Forest** model (Model 2) improved performance by reducing variance through ensemble learning but still struggled with image data complexity. 
- A baseline **Convolutional Neural Network (CNN)** (Model 3) significantly outperformed these models by capturing spatial features, though it faced overfitting issues.
![CNN Model Accuracy](/Graphs/cnn_modelacc.jpg)

- Finally, a **Hyperparameter-Tuned CNN** achieved the best results, with optimized parameters and regularization methods like dropout, leading to higher accuracy, precision, and recall, and better generalization on unseen data. Resource on['Hyperparameter Tuning'](https://www.geeksforgeeks.org/hyperparameter-tuning/)
# Evaluation:
The final model, optimized through hyperparameter tuning, achieved a test accuracy of 99.92%. However, this high accuracy may be unreliable due to potential overfitting and the lack of downscaling of face images, suggesting the model might be leveraging size differences rather than meaningful features. Accuracy was the primary evaluation metric, but the near-perfect score indicates overfitting, implying poor generalization to new data. The hyperparameter tuning and model training took 1 hour, 20 minutes, and 10 seconds. Despite efficient resource use, further steps are needed to ensure the model's reliability and generalization.
![Confusion Matrix of Best Model](/Graphs/confusion_matrix.jpg)

# Conclusion:
The hyperparameter-tuned model achieved outstanding results, with a test accuracy of 99.92%. The classification report further confirmed this performance, showing perfect precision, recall, and F1-scores of 1.00 for both classes (faces and non-faces). The confusion matrix also reflected this high accuracy, with no misclassifications. These results indicate that the model performs exceptionally well in distinguishing between faces and non-faces, although the high accuracy suggests potential overfitting due to the image size discrepancy between the datasets.

## Limitations:
The project faced several limitations. The significant size difference between face images from the LFW dataset (approximately 8k) and object images from the Tiny ImageNet dataset (around 1.5k) likely contributed to overfitting, as the model might distinguish based on size rather than features. There was also a dataset imbalance, leading to biased performance favoring the overrepresented class. The near-perfect accuracy indicated overfitting, with the model performing well on training data but poorly on new data. Variations in image quality, lighting, and backgrounds affected performance, making the model sensitive to these inconsistencies. Additionally, the current model architecture might not be optimal for distinguishing faces from non-faces, suggesting a need for more advanced feature extraction techniques or different architectures.

## Next Steps:
To improve the model, the next steps include downscaling face images to match object image sizes for uniformity, applying data augmentation techniques to increase training data diversity, and implementing regularization methods like dropout and L2 regularization to prevent overfitting. Additionally, experimenting with advanced architectures such as deeper CNNs or transfer learning can enhance feature extraction, while using cross-validation will better assess performance and ensure generalization. These steps aim to enhance the model's robustness and accuracy without overfitting.

## More Information:
Find the full analysis in the [Notebook](Facial_Recognition_using_CNN.ipynb) or review this [presentation](FacialRecognitionClassifier.pdf).

## Repository Structure
- Graphs
- README.md
- Facial_Recognition_using_CNN.ipynb
- FacialRecognitionClassifier.pdf

### Author: Adriana Ramirez Franco (aramirezfr20@gmail.com)
