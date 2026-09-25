# Capstones

### 1. Fraudulent Transactions Detection (Status: Re-evaluated, see Version 3)
Leveraging a combination of statistical analysis and ML algorithms by analyzing the anomaly within the historical data and identifying the fraudulent transactions. 
The reason for sharing this project is because it inspired creative ideas, offering insights into handling complex data and figuring out effective solutions. 
It allowed me to explore various approaches to address imbalanced data, and implement lesser-known Python libraries, adding significant value to this project.

**Update (Version 3):** the ~98% scores in Versions 1 and 2 came from oversampling before the train/test split. Re-evaluated with a leak-safe pipeline, no model beats chance (AUC ≈ 0.50), and `TransactionKey` turned out to be a perfect label leak. Version 3 is a reusable pipeline plus the evidence; see [`Fraud-Detection/Version 3`](Fraud-Detection/Version%203/README.md).

<br>

### 2. American Sign Language Detection (Status: In-Progress)
Developed an ASL detection model that aims to recognize sign language gestures using hand movements through a Convolutional Neural Network (CNN). Trained on Kaggle's P100 GPUs, it reaches 98.8% on a random hold-out split of the Kaggle dataset (91% training accuracy). That split is optimistic: the images are near-duplicate frames from one room, and it was also used for early stopping. Tools for testing on your own webcam images are now included; see [`ASL Detection`](ASL%20Detection/README.md). With the right training data, this model has the potential to adapt to other sign languages as well.

##### Next Steps:
- Real-World Testing: Score the model on my own webcam captures (`webcam_demo.py --collect`, `evaluate.py`) and replace the headline number.
- Fine-Tuning: Retrain with the block-split, augmented `train.py` for better generalisation.
- Real-Time Recognition: A live webcam demo exists (`webcam_demo.py`); crop to the hand first for robustness.
- User Interface: Developing a user-friendly interface to demonstrate the model.

Stay tuned for exciting updates as I continue to refine and expand this project!

<br>

### 3. [Spotify and Youtube Analysis, via Power-BI (Status: Completed)](https://www.kaggle.com/datasets/salvatorerastelli/spotify-and-youtube)
Created a Power BI project analyzing Spotify and YouTube using a limited dataset to gain insights into the music landscape. The objective was to determine which platform should lead in streaming music based on various metrics and insights.

##### Purpose:
- Gain Insights: Understand music trends and user preferences.
- Platform Decision: Identify whether Spotify or YouTube should lead the music streaming industry.

The project is complete, providing valuable insights for strategic decisions in the music streaming industry.






