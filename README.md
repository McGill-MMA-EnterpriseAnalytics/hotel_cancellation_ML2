## Predicting Hotel Booking Cancellations

Final Project for INSY695 Enterprise ML-2.


**Group Members**
* Alisa Liu
* Chiara Lu
* Hao Duong
* Jaya Chaturvedi
* Keani Schuller
* Reo Paul Jackson


## Introduction


This project focuses on predicting hotel booking cancellations at the time of reservation. Our model assists hoteliers in forecasting the likelihood of cancellations, thereby optimizing revenue through dynamic pricing strategies, overbooking management, and targeted marketing efforts. The goal of this project is to enhance hotel revenue management and guest experience by providing reliable predictions on reservation cancellations. This allows for effective strategic decisions in pricing and booking policies.


## Hyperparameter Tuning
To improve the model created in the first ML course, we employed several optimization techniques, such as Bayesian Optimization, TPE (Tree-structured Parzen Estimator), RandomizedSearchCV, and GridSearchCV. The fine-tuned parameters significantly improved the model's performance compared to the base model, increasing the ROC-AUC score from 86.47% to 94.12%.

The parameters of our final model are:
* Algorithm: Random Forest
* max_features: None
* min_samples_leaf: 1
* n_estimators: 200
* ROC-AUC Score: 94.12% with manual tuning


## AutoML
On top of using manual hyperparameter tuning techniques, we also tested AutoML models to evaluate their scores in comparison to the base Random Forest from the previous project. The benefits of using AutoML is that is it more efficient than manual techniques, it is easily accessible to all, and that its results are easily reproduceable. The three libraries used were H2O.ai, TPOT, and Lale. In addition, three TPOT models were also tried. The results are as follows:
* H20.ai and Lale performed slightly worse than our base model with ROC-AUC scores of 85.66% and 84.43% respectively.
* All three TPOT models performed better than our base model but worse than the best tuned Random Forest, with the following scores:
  * TPOT-nn: 92.84%
  * TPOT-Light: 89.56%
  * TPOT-MDR: 88.95%
    
<img width="472" alt="image" src="https://github.com/McGill-MMA-EnterpriseAnalytics/hotel_cancellation_ML2/assets/91162706/347cbc4f-abf3-4456-a522-c3bc509f15a1">

## Deployment 
The model is deployed using Docker containers locally and on the cloud via Databricks, ensuring flexibility and scalability. The Docker deployment process used started by pickling the final model into a pickle file. Next, FastAPI was used for creating prediction endpoints for both single and batch processing. A Gradio web application was created as a user interface to allow the user to make predictions on new inputs. Docker Image was used for files and dependencies, and Docker containers were used to then run the FastAPI & Gradio applications. 


## Model Fairness and Explainability
For fairness, we evaluated the final model using FairML, ensuring decisions do not disproportionately impact specific groups. The results for the FairML audit can be found below. The results suggest that certain variable disproportionately impact the model to predict a cancellation (positive influences) or not (negative influences). The results included the following insights:
* market_segment_Online TA (Online Travel Agent) has the highest positive influence, which means reservations made through online travel agents significantly increase the likelihood that the model predicts a cancellation 
* lead_time has the highest negative influence, which means reservations made well in advance decrease the likelihood it predicts a cancellation
* Not surprisingly, deposit_type_Non_Refund, or booked reservations with no deposit refunds, decrease the likelihood of a cancellation prediction


For explainability, we utilized LIME to interpret the model's predictions. This provides insights into the results of the model, such understanding the chances that the model predicts a cancellation and the factors influencing the likelihood of cancellations. The results included the following insights:
* The model predicts that there's a 77% chance that the reservation will not be cancelled and a 23% chance that it will
* The arrival month value of 0.59 suggests that the month of the arrival date contributes towards a higher chance of cancellation
* Variables like children, babies and is_repeated_guest all have negative values, meaning they contribute to a lesser likelihood of cancellation


## Model Drift
We used two types of drift evaluation: adversarial drift and model drift using KSDrift. These drifts were evaluated using differences in train and test set predictions. In both types of drift evaluated, the drift evaluated was zero. This means the characteristics of the distributions evaluated are the same. 


## MLOps Best Practices Used
The Project Management style used to manage the project work was Kanban methodology. This meant we worked in small teams to tackle major problems in small steps. To ensure all experiments were properly logged and monitored, we used MLflow to keep track of all the experimentation done. We also made sure to saved all our failed experiments as well. For version control, we used GitLens to allow us to review any changes made on GitHub.


## Conclusion
This project successfully integrates advanced ML techniques to provide a robust solution for predicting hotel booking cancellations. It offers significant value to hoteliers by enabling more informed decision-making and optimizing various aspects of hotel management.
