## Running the project

1. You need `uv` installed.
2. Clone the git repo
3. Navigate to the project directory and run the following commands:

```bash
uv venv

source .venv/bin/activate # for windows use .venv\Scripts\activate

uv pip install -r pyproject.toml

uv run uvicorn main:app --reload
```

4. The server will start on `http://localhost:8000`
5. You can use `curl` or any other HTTP client to test the endpoints.

## Endpoints

1. `/upload` (POST):

   - Accepts a CSV file upload
   - Saves it as `data.csv` in the project directory
   - Returns a success message or error if something goes wrong

2. `/train` (POST):

   - Requires a password parameter `sec` for authentication
   - Loads the data from `data.csv`
   - Preprocesses the data using the same steps from your notebook
   - Creates sequences for LSTM training
   - Builds and trains the LSTM model
   - Saves the trained model and scaler for later use
   - Returns training metrics (accuracy, precision, recall)

3. `/predict` (POST):

   - Accepts a location name and optionally the number of days of historical data to use
   - Loads the trained model and scaler
   - Gets the latest data for the specified location
   - Generates a 7-day forecast
   - Returns predictions with dates and confidence scores

4. `/addRec` (POST):
   - Adds a new weather record to the dataset
   - Accepts a JSON payload with the following required fields:
     - `Date`: Date in YYYY-MM-DD format
     - `Location`: Location name
     - `MinTemp`: Minimum temperature
     - `MaxTemp`: Maximum temperature
     - `Rainfall`: Rainfall amount
     - `WindGustDir`: Wind gust direction (N, NNE, NE, etc.)
     - `WindGustSpeed`: Wind gust speed
     - `WindDir9am`, `WindDir3pm`: Wind directions at 9am and 3pm
     - `WindSpeed9am`, `WindSpeed3pm`: Wind speeds at 9am and 3pm
     - `Humidity9am`, `Humidity3pm`: Humidity levels (0-100)
     - `Pressure9am`, `Pressure3pm`: Pressure readings
     - `Cloud9am`, `Cloud3pm`: Cloud cover (0-9)
     - `Temp9am`, `Temp3pm`: Temperatures at 9am and 3pm
     - `RainToday`: "Yes" or "No"
   - Optional fields:
     - `Evaporation`
     - `Sunshine`
     - `RainTomorrow`
   - Performs validations on the data (e.g., MaxTemp > MinTemp)
   - Returns success message with the added record

To use these endpoints:

1. First, upload your data:

```bash
curl -X POST -F "file=@weatherAUS.csv" http://localhost:8000/upload
```

2. Train the model (replace YOUR_PASSWORD with the actual password - currently `aaaa54121`):

```bash
curl -X POST "http://localhost:8000/train?sec=YOUR_PASSWORD"
```

3. Get predictions for a location:

```bash
curl -X POST "http://localhost:8000/predict?location=Sydney"
```

4. Add a new weather record:

```bash
curl -X POST "http://localhost:8000/addRec" \
  -H "Content-Type: application/json" \
  -d '{
    "Date": "2024-02-20",
    "Location": "Sydney",
    "MinTemp": 15.0,
    "MaxTemp": 25.0,
    "Rainfall": 0.0,
    "WindGustDir": "SE",
    "WindGustSpeed": 35.0,
    "WindDir9am": "E",
    "WindDir3pm": "SE",
    "WindSpeed9am": 10.0,
    "WindSpeed3pm": 15.0,
    "Humidity9am": 75.0,
    "Humidity3pm": 60.0,
    "Pressure9am": 1015.0,
    "Pressure3pm": 1013.0,
    "Cloud9am": 4.0,
    "Cloud3pm": 6.0,
    "Temp9am": 18.0,
    "Temp3pm": 23.0,
    "RainToday": "No"
  }'
```

The prediction response will look like this:

```json
{
	"location": "Sydney",
	"forecast": [
		{
			"date": "2024-02-20",
			"rain_predicted": true,
			"confidence": 0.85
		}
	]
}
```

![image](https://github.com/user-attachments/assets/935a0cb5-fab4-46d4-8f0f-b8def112c714)
