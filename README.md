1. `/upload` (POST):

   - Accepts a CSV file upload
   - Saves it as `data.csv` in the project directory
   - Returns a success message or error if something goes wrong

2. `/train` (POST):

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

To use these endpoints:

1. First, upload your data:

```bash
curl -X POST -F "file=@weatherAUS.csv" http://localhost:8000/upload
```

2. Train the model:

```bash
curl -X POST http://localhost:8000/train
```

3. Get predictions for a location:

```bash
curl -X POST "http://localhost:8000/predict?location=Sydney"
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
