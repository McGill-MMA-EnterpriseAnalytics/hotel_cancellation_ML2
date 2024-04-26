## Predicting Hotel Booking Cancellations

Final Project for INSY695 EML-2


**Group Members**
* Alisa Liu
* Chiara Lu
* Hao Duong
* Jaya Chaturvedi
* Keani Schuller
* Reo Paul Jackson


### Introduction


This project focuses on predicting hotel booking cancellations at the time of reservation. Our model assists hoteliers in forecasting the likelihood of cancellations, thereby optimizing revenue through dynamic pricing strategies, overbooking management, and targeted marketing efforts. The goal of this project is to enhance hotel revenue management and guest experience by providing reliable predictions on reservation cancellations. This allows for effective strategic decisions in pricing and booking policies.


### Hyperparameter Tuning
To improve the model created in the first ML course, we employed several optimization techniques, such as Bayesian Optimization, TPE (Tree-structured Parzen Estimator), RandomizedSearchCV, and GridSearchCV. The fine-tuned parameters significantly improved the model's performance compared to the base model, increasing the ROC-AUC score from 86.47% to 94.12%.

The parameters of our final model are:
Algorithm: Random Forest
max_features: None
min_samples_leaf: 1
n_estimators: 200
ROC-AUC Score: 94.12% with manual tuning


### Deployment 
The model is deployed using Docker containers locally and on the cloud via Databricks, ensuring flexibility and scalability. The Docker deployment process used started by pickling the final model into a pickle file. Next, FastAPI was used for creating prediction endpoints for both single and batch processing. A Gradio web application was created as a user interface to allow the user to make predictions on new inputs. Docker Image was used for files and dependencies, and Docker containers were used to then run the FastAPI & Gradio applications. 


### Model Fairness and Explainability
For fairness, we evaluated the final model using FairML, ensuring decisions do not disproportionately impact specific groups. The results for the FairML audit can be found below. The results suggest that certain variable disproportionately impact the model to predict a cancellation (positive influences or red bars) or not (negative influences or blue bars). 

For explainability, we utilized LIME to interpret the model's predictions. This provides insights into the results of the model, such understanding the chances that the model predicts a cancellation and the factors influencing the likelihood of cancellations.


### MLOps Best Practices Used
The Project Management style used to manage the project work was Kanban methodology. This meant we worked in small teams to tackle major problems in small steps. To ensure all experiments were properly logged and monitored, we used MLflow to keep track of all the experimentation done. We also made sure to saved all our failed experiments as well. For version control, we used GitLens to allow us to review any changes made on GitHub.


### Conclusion
This project successfully integrates advanced ML techniques to provide a robust solution for predicting hotel booking cancellations. It offers significant value to hoteliers by enabling more informed decision-making and optimizing various aspects of hotel management.
