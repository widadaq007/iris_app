# knn_api.py
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Train the KNN model
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Save the trained model
with open("knn_model.pkl", "wb") as f:
    pickle.dump(knn_model, f)

# Load the trained model
with open("knn_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize FastAPI
app = FastAPI()

# HTML form
html_form = """
<!DOCTYPE html>
<html>
<head>
    <title>KNN Iris Prediction</title>
</head>
<body>
    <h2>Enter Iris Flower Measurements:</h2>
    <form action="/predict" method="post">
        Sepal Length: <input type="number" step="0.1" name="sepal_length"><br><br>
        Sepal Width: <input type="number" step="0.1" name="sepal_width"><br><br>
        Petal Length: <input type="number" step="0.1" name="petal_length"><br><br>
        Petal Width: <input type="number" step="0.1" name="petal_width"><br><br>
        <input type="submit" value="Predict">
    </form>
</body>
</html>
"""

# Home page
@app.get("/", response_class=HTMLResponse)
async def home():
    return html_form

# Prediction route
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction_index = model.predict(input_data)[0]
    predicted_class = iris.target_names[prediction_index]
    
    # Return HTML with prediction
    return f"""
    <!DOCTYPE html>
    <html>
    <head><title>Prediction Result</title></head>
    <body>
        <h2>Prediction Result</h2>
        <p>Predicted Iris Class: <b>{predicted_class}</b></p>
        <a href="/">Go Back</a>
    </body>
    </html>
    """
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)