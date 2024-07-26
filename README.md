# DSP_ML_Model: Predict Using FASTAPI Request
## Hosted on 
```
https://dps-ml-model.onrender.com/
```
predict go to predict sunroute

```
https://dps-ml-model.onrender.com/predict
```

## Setup Instructions

### 1. Pull the Repository

First, clone the repository to your local machine using the command:

```
git clone https://github.com/pratim4dasude/dps_ml_model
```
### 2. Install Dependencies
Navigate to the repository directory and install the required dependencies listed in requirements.txt:

```
cd dps_ml_model
pip install -r requirements.txt
```
### 3. Run the Model
To run the model, execute the following script:

```
python main_model.py
```
This will generate the model.pkl file.

### 4. Start FastAPI Server
Start the FastAPI server using the command:
```
fastapi dev main.py
```
### 5. Access the API
Click on the local host link provided in the terminal to access the site.
```
http://127.0.0.1:8000  
```
### 6. Test the API with Postman
To test the POST request, you can use Postman:

## Open Postman.
Select POST and enter the localhost URL followed by /predict.
Go to Body -> raw and select JSON from the dropdown menu.
Enter the following JSON data:
```
{
    "Category": "Alkoholunfälle",
    "Accident_type": "insgesamt",
    "Year": 2021,
    "Month": 5
}
```
Click Send at the right-hand corner.
You will see the output JSON which looks like this:

```
{
    "prediction": 141.41836547851562
}
```
Congratulations! You've successfully run the model and tested the API.
