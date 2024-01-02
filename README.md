# Project Name

## Tweet Emotion and and what product/company it is directed at

### Overview

This project aims to address missing values in the columns "emotion_in_tweet_is_directed_at" and “is_there_an_emotion_directed_at_a_brand_or_product” in a given dataset. The missing values are imputed and predicted using fine-tuned pre-trained transformer models. The models utilized include "bert-base-uncased" for emotion text classification and "dslim/bert-base-NER" for named entity recognition (NER) and token classification.

### Models Selection

The choice of models is based on the popularity and performance of pre-trained transformer models available in the Hugging Face model library. Specifically, "bert-base-uncased" is used for emotion classification, and "dslim/bert-base-NER" is used for NER tasks. These models have been fine-tuned to better suit the dataset and the objectives of the project.

### Data Augmentation Techniques

To enhance the dataset and improve model generalization, various data augmentation techniques have been applied:

1. **Synonym Replacement:** Replaces words with their synonyms.
2. **Back Translation:** Translates the text to another language and then back to the original language.
3. **Random Insertion:** Randomly inserts new words into the text.
4. **Random Deletion:** Randomly deletes words from the text.
5. **Random Swap:** Swaps words randomly in the text.
6. **Contractions Expansion:** Expands contractions in the text.
7. **Sentence Splitting:** Splits sentences into multiple shorter sentences.
8. **Emotion Flipping:** Inverts the emotion in a sentence.

### Handling Missing Values

1. **Imputation for "emotion_in_tweet_is_directed_at":**
   - As this column has a significant number of missing values, imputation is performed by filling missing values with the mode.

2. **Prediction for "is_there_an_emotion_directed_at_a_brand_or_product":**
   - Utilizes the fine-tuned "bert-base-uncased" model for emotion text classification to predict missing values.

3. **Predicting Brands/Products Mentioned in Emotions:**
   - Employs the NER model "dslim/bert-base-NER" to predict organizations, companies, or brands mentioned in the dataset.

### Data Preprocessing and Oversampling

To address computational constraints during model training, data preprocessing techniques such as oversampling through synonym replacement and batch processing have been applied. These steps ensure efficient training of the transformer models.

### Model Performance

- **Emotion Classifier:**
  - Achieves an accuracy of 80% and above on the tweet_text column.

- **NER Model:**
  - Uses Seqeval as a metric for named entity recognition, evaluating the model's ability to label sequences. Seqeval is well-suited for NLP tasks like NER.

- Write about the requirements and ways to deploy the model to prod. Also explain about different types of metrics someone should monitor after deployment.
1. Select a deployment platform based on scalability and resource requirements (e.g., AWS, Azure, GCP), on-premises servers, or container orchestration platforms. Here, I would pefer AWS Sagemaer and ml.p2.xlarge for the resource requirements, since it requires high computational resources.
2. Save the fine-tuned model in a format suitable for deployment, such as ONNX, TensorFlow SavedModel, or PyTorch JIT format.
3. Implement logging mechanisms to capture model inputs, outputs, and errors. Establish monitoring for real-time tracking of model performance, resource utilization, and potential issues.
4. Monitor key performance metrics such as response time, throughput, and resource utilization. Track these metrics to ensure the model operates within acceptable performance thresholds.
5. Regularly assess the model for drift by comparing the production data distribution with the training data distribution. We need to implement strategies to retrain or update the model when significant drift is detected.

### Conclusion

The project successfully imputes missing values in the dataset, predicts emotions and brands/products mentioned in tweets, and achieves reliable model performance on various metrics. The combination of fine-tuned transformer models and data augmentation techniques contributes to the overall effectiveness of the solution. The project addresses both missing data challenges and enhances the dataset for downstream analysis and applications.
