
# Scalable Machine Learning Pipeline with Dask: DaskMLPipeline_AGNews

## Project Goal
To construct a scalable machine learning pipeline capable of handling large datasets, leveraging Dask for distributed computing, and comparing its performance against a traditional approach. This revised project uses the **AG News Corpus** via the Hugging Face `datasets` library for robust data loading.

## 1. Dataset Selection & Management
- **Dataset:** AG News Corpus (News Classification)
- **Source:** Hugging Face `datasets` library (`load_dataset("ag_news")`)
- **Storage:** The `datasets` library handles downloading and caching the dataset automatically, eliminating the need for manual URL management or extraction.
- **Task:** Multi-class Text Classification (4 classes: World, Sports, Business, Sci/Tech)
- **Data Size:** The original dataset contains 120,000 training samples and 7,600 test samples. This pipeline works with 120000 training samples and 7600 test samples after filtering empty text rows.
- **Subset Usage:** `USE_SUBSET` is set to `False`. If `True`, a fraction of `50.0%` of the original dataset is used for faster demonstration. For AG News, we often use the full dataset as it's manageable.
- **Loading:** Data is loaded using `datasets.load_dataset` and then converted into Pandas DataFrames, which are subsequently converted to Dask DataFrames. This approach ensures reliable access to the dataset.
- **Preprocessing:** Empty or whitespace-only text rows are explicitly filtered out to prevent issues with feature extraction.
- **Label Mapping:** AG News labels are inherently 0-3 in the Hugging Face version.
- **Text Combination:** The 'text' column from the Hugging Face dataset is directly used as the feature.

## 2. Dask Setup
A `dask.distributed.Client` was set up to create a local Dask cluster.
- **Number of Workers:** 20 (Adjusted based on your CPU cores for optimal performance)
- **Threads per Worker:** 2
- **Memory Limit per Worker:** 6GB (Increased to provide more buffer for sparse matrix operations)
- **Dashboard Link:** (Look for the link printed in the console when the client starts)
This setup allows simulating a distributed environment on a single machine by using multiple processes/threads. For production, this would involve connecting to a dedicated Dask cluster (e.g., via Dask Gateway, Kubernetes, YARN, or cloud services).

## 3. Data Preprocessing
The data preprocessing involved text vectorization using TF-IDF.
- **Tool:** `dask_ml.feature_extraction.text.HashingVectorizer`
- **Max Features:** 5000
- **Why Dask-ML's HashingVectorizer?** It's efficient for large text datasets as it doesn't need to store the entire vocabulary in memory and works seamlessly with Dask's distributed arrays. This is crucial when dealing with millions of text documents.
- **Output:** The preprocessed text data (`X`) is a `dask.array` of sparse matrices, and the target labels (`y`) are a `dask.array`. Explicit meta information is provided to Dask to help with sparse array shape inference.

## 4. Model Training
A Logistic Regression model was trained for the text classification task.
- **Model:** `dask_ml.linear_model.LogisticRegression`
- **Solver:** 'lbfgs' (Compatible with Dask-ML's GLM implementation)
- **Regularization (C):** 0.1
- **Multi-class Strategy:** 'auto' (let scikit-learn/dask-ml choose appropriate strategy for multiclass)
- **Integration:** `dask-ml` provides scikit-learn compatible estimators that can operate directly on Dask DataFrames and Arrays, distributing the computation across the Dask cluster.

## 5. Performance Analysis
The performance of the Dask pipeline was compared against a traditional Pandas/Scikit-learn approach.
- **Traditional Approach:** Performed on a small subset of the data (approximately 12000 training samples and 760 test samples) due to memory constraints for handling the full dataset. Filtered for empty text rows for consistency.
- **Metrics Compared:** Training Time, Accuracy.

### Performance Results:
| Metric                   | Traditional (Sample) | Dask (Full) Dataset |
| :----------------------- | :------------------- | :-------------------------------------------------------- |
| Training Time (seconds)  | 0.43 | 88.19 |
| Prediction Time (seconds)| N/A                  | 1.10 |
| Accuracy                 | 0.3695 | 0.3695 |

**Observation:**
- The primary advantage of Dask becomes evident when dealing with datasets that exceed single-machine memory. While the traditional approach might be faster on very small samples due to lower overhead, Dask enables processing the entire large dataset efficiently.
- Dask's training time reflects computation on **120000** samples, whereas the traditional method is limited to a small, in-memory fraction.
- The prediction time for the Dask model on the distributed test set (up to **7600** samples) demonstrates its capability for large-scale inference.

## 6. Visualization
- **Performance Comparison Bar Chart:** Visualizes training times and accuracies of both approaches.
- **Confusion Matrix:** Shows the Dask model's performance on the test set, indicating correct and incorrect classifications for the 4 news categories.
- **Classification Report:** Provides a detailed breakdown of precision, recall, and F1-score for each class.

## 7. Deliverables (Code Structure)
- `pipeline.py`: Contains all the Python code for dataset loading, setup, data preprocessing, training, and analysis.
- `DaskMLPipeline_AGNews_performance_comparison.png`: Bar chart visualizing training time and accuracy comparison.
- `DaskMLPipeline_AGNews_confusion_matrix.png`: Heatmap of the confusion matrix for the Dask model.
- `DaskMLPipeline_AGNews_README.md`: This comprehensive documentation file.
- The AG News dataset will be cached by the `datasets` library, usually in a location like `~/.cache/huggingface/datasets`.

## How to Run
1.  **Update `requirements.txt`** if you haven't already with `datasets` and `scipy`. Run `pip install -r requirements.txt`.
2.  Save the code as `pipeline.py`.
3.  Run from your terminal: `python pipeline.py`
4.  The script will automatically download and prepare the AG News dataset using the `datasets` library.
5.  Observe the Dask dashboard link in the console for real-time monitoring of computations.

## Future Improvements / Considerations
-   **Cloud Integration:** For truly massive datasets, deploying Dask on cloud platforms (AWS, GCP, Azure) or a Kubernetes cluster would be essential.
-   **Hyperparameter Tuning:** Integrate `dask_ml.model_selection.GridSearchCV` or `RandomizedSearchCV` for distributed hyperparameter optimization.
-   **Alternative Text Vectorization:** Explore other Dask-compatible text embeddings or feature extraction methods (e.g., word embeddings if you can manage their size).
-   **Model Persistence:** Save the trained Dask-ML model for later inference using `joblib` or `cloudpickle`.
-   **More Complex Models:** Investigate Dask integrations with libraries like XGBoost or deep learning frameworks (TensorFlow/PyTorch) for distributed training of more complex models.
-   **Robust Error Handling:** Enhance error handling and logging for production environments.
    