import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt

def load_data(file_path):
    return pd.read_excel(file_path)

def preprocess_data(data):
    # Backup original flight info
    original_data = data.copy()

    # Drop unnecessary column
    data.drop(['Unnamed: 0', 'flight'], axis=1, inplace=True)

    # Encode categorical features
    data_encoded = pd.get_dummies(data, columns=['airline','source_city','departure_time','stops','arrival_time','destination_city','class'], drop_first=True)

    x = data_encoded.drop('price', axis=1)
    y = data_encoded['price']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = LinearRegression()
    model.fit(x_train_scaled, y_train)

    joblib.dump(model, r'E:\Python\internship_of_Ai\Price_prediction\linear_regression_model.pkl')

    y_pred = model.predict(x_test_scaled)

    return model, scaler, x_test_scaled, y_test, x_test, y_pred, original_data.iloc[x_test.index]

def save_outputs(x_test_original, y_test, y_pred, x_test_encoded):
    # Combine raw and prediction data
    results_df = x_test_original.copy()
    results_df['Actual_Price'] = y_test.values
    results_df['Predicted_Price'] = y_pred
    results_df.reset_index(drop=False, inplace=True)
    results_df.rename(columns={'index': 'Record_ID'}, inplace=True)

    # Metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    metrics_df = pd.DataFrame({
        'Metric': ['R-squared', 'Mean Absolute Error'],
        'Value': [r2, mae]
    })

    # Export to Excel
    output_file = r'E:\Python\internship_of_Ai\Price_prediction\model_results.xlsx'
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        results_df.to_excel(writer, sheet_name='Prediction_Results', index=False)
        metrics_df.to_excel(writer, sheet_name='Model_Metrics', index=False)
        x_test_encoded.to_excel(writer, sheet_name='Encoded_Features', index=False)

    print(f"Results saved to: {output_file}")
    print(f"R²: {r2:.4f}, MAE: {mae:.2f}")
    return r2, mae

# def plot_results(y_test, y_pred):
#     plt.figure(figsize=(8,6))
#     plt.scatter(y_test, y_pred, alpha=0.5, color='dodgerblue')
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
#     plt.xlabel('Actual Price')
#     plt.ylabel('Predicted Price')
#     plt.title('Actual vs Predicted Flight Prices')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

def main():
    file_path = r'E:\Python\internship_of_Ai\Price_prediction\Clean_Dataset.xlsx'
    data = load_data(file_path)
    print("✅ Data loaded successfully")

    model, scaler, x_test_scaled, y_test, x_test_encoded, y_pred, x_test_original = preprocess_data(data)
    print("✅ Model trained successfully")

    r2, mae = save_outputs(x_test_original, y_test, y_pred, x_test_encoded)
    # plot_results(y_test, y_pred)

    print(f"\n📊 R-squared: {r2:.4f}")
    print(f"📉 Mean Absolute Error: {mae:.2f}")

if __name__ == "__main__":
    main()
