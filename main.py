from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
from datetime import datetime, timedelta, timezone

app = Flask(__name__)
CORS(app)

# --- Konfigurasi Global, Pemuatan Model, dan Logging Sederhana ---
MODELS = {}
AVAILABLE_EMOTIONS = ['angry', 'fear', 'happy', 'sad', 'surprised']
MODEL_PATH_TEMPLATE = 'models/model_prophet_{emotion}.pkl'

# Sistem log sederhana berbasis memori
API_LOGS = []

def add_log(level, message):
    """Menambahkan entri log ke daftar API_LOGS."""
    timestamp = datetime.now(timezone.utc).isoformat()
    API_LOGS.append({"timestamp": timestamp, "level": level, "message": message})
    # Batasi jumlah log agar tidak terlalu besar di memori (opsional)
    if len(API_LOGS) > 200:
        API_LOGS.pop(0)

add_log("INFO", "API Flask mulai dijalankan.")

for emotion in AVAILABLE_EMOTIONS:
    try:
        with open(MODEL_PATH_TEMPLATE.format(emotion=emotion), 'rb') as f:
            MODELS[emotion] = pickle.load(f)
        add_log("INFO", f"Model untuk emosi '{emotion}' berhasil dimuat dari {MODEL_PATH_TEMPLATE.format(emotion=emotion)}.")
    except FileNotFoundError:
        add_log("WARNING", f"File model untuk emosi '{emotion}' tidak ditemukan di {MODEL_PATH_TEMPLATE.format(emotion=emotion)}.")
    except Exception as e:
        add_log("ERROR", f"Gagal memuat model untuk emosi '{emotion}'. Error: {e}")

# --- Fungsi Helper ---
def get_forecast_accuracy(model, forecast_df_full):
    if not hasattr(model, 'history') or model.history.empty:
        add_log("WARNING", f"Model '{model.model_name if hasattr(model, 'model_name') else 'emosi tidak diketahui'}' tidak memiliki histori untuk perhitungan akurasi.")
        return 0
    merged_df = pd.merge(forecast_df_full[['ds', 'yhat']], model.history[['ds', 'y']], on='ds', how='inner')
    if merged_df.empty or 'yhat' not in merged_df.columns or 'y' not in merged_df.columns: return 0
    y_actual, y_hat_hist = merged_df['y'], merged_df['yhat']
    if len(y_actual) == 0: return 0
    valid_actuals = y_actual[y_actual != 0]
    if valid_actuals.empty: return 100.0 if (y_hat_hist[y_actual == 0] == 0).all() else 0.0
    mape = (abs(y_hat_hist[y_actual != 0] - valid_actuals) / valid_actuals).mean() * 100
    accuracy = round(100 - mape, 2) if not pd.isna(mape) else 0
    return max(0, min(100, accuracy))

def get_weekly_trend(forecast_df_full): # Diganti nama untuk kejelasan
    if len(forecast_df_full) < 14: return 'N/A'
    last_week_yhat, prev_week_yhat = forecast_df_full.tail(7)['yhat'], forecast_df_full.tail(14).head(7)['yhat']
    if last_week_yhat.isnull().all() or prev_week_yhat.isnull().all(): return 'N/A'
    last_week_mean, prev_week_mean = last_week_yhat.mean(), prev_week_yhat.mean()
    if pd.isna(last_week_mean) or pd.isna(prev_week_mean): return 'N/A'
    if prev_week_mean == 0: return '+0.00%' if last_week_mean >= 0 else '-0.00%'
    delta = ((last_week_mean - prev_week_mean) / prev_week_mean) * 100
    return f"{'+' if delta >= 0 else ''}{round(delta, 2)}%"

# --- Endpoint API ---
@app.route('/', methods=['POST'])
def get_forecast_data():
    add_log("INFO", f"Menerima permintaan POST ke '/' dengan payload: {request.json}")
    payload = request.get_json()
    if not payload: 
        add_log("WARNING", "Permintaan ke '/' tanpa payload JSON.")
        return jsonify({"error": "Request payload tidak valid"}), 400
        
    selected_emotion = payload.get('emotion', 'happy')
    if selected_emotion not in MODELS: 
        add_log("ERROR", f"Model untuk emosi '{selected_emotion}' tidak ditemukan saat request ke '/'.")
        return jsonify({"error": f"Model '{selected_emotion}' tidak ditemukan"}), 404
        
    model = MODELS[selected_emotion]
    if not hasattr(model, 'history') or model.history.empty: 
        add_log("ERROR", f"Model '{selected_emotion}' tidak memiliki data histori saat request ke '/'.")
        return jsonify({"error": f"Model '{selected_emotion}' tidak punya histori."}), 500

    forecast_days_count = int(payload.get('forecast_days', 7))
    granularity_frontend = payload.get('granularity', 'daily')
    prophet_freq = {"hourly": "H", "daily": "D", "weekly": "W", "monthly": "M"}.get(granularity_frontend.lower(), 'D')
    start_date_str, end_date_str = payload.get('start_date'), payload.get('end_date')

    future_df = model.make_future_dataframe(periods=forecast_days_count, freq=prophet_freq)
    forecast_df_full = model.predict(future_df)
    output_df = pd.merge(forecast_df_full[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], model.history[['ds', 'y']], on='ds', how='left')

    if start_date_str and end_date_str:
        try:
            start_dt_filter, end_dt_filter = pd.to_datetime(start_date_str), pd.to_datetime(end_date_str)
            output_df = output_df[(output_df['ds'] >= start_dt_filter) & (output_df['ds'] <= end_dt_filter)]
        except Exception as e: add_log("ERROR", f"Error parsing tanggal filter di '/': {e}")

    recharts_data_list = []
    if not output_df.empty:
        recharts_data = output_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'y']].copy()
        recharts_data.rename(columns={'y': 'actual', 'ds': 'name'}, inplace=True)
        recharts_data['name'] = recharts_data['name'].dt.strftime('%Y-%m-%d')
        for col in ['actual', 'yhat_lower', 'yhat_upper', 'yhat']:
            if col in recharts_data.columns:
                 recharts_data[col] = recharts_data[col].apply(lambda x: None if pd.isna(x) else x)
        recharts_data_list = recharts_data.to_dict(orient='records')
    
    add_log("INFO", f"Berhasil memproses permintaan ke '/' untuk emosi '{selected_emotion}'. Mengembalikan {len(recharts_data_list)} poin data.")
    return jsonify({
        'forecast_points': recharts_data_list,
        'accuracy': get_forecast_accuracy(model, forecast_df_full),
        'trend': get_weekly_trend(forecast_df_full), # Ini adalah tren mingguan
        'emotion': selected_emotion, 'forecast_days': forecast_days_count, 'granularity': granularity_frontend
    })

@app.route('/api/summary', methods=['GET'])
def api_summary():
    add_log("INFO", "Menerima permintaan GET ke '/api/summary'.")
    summary_forecast_days, summary_freq = 7, 'D'
    accuracies, trends = {}, {}
    for emotion_key in AVAILABLE_EMOTIONS:
        if emotion_key not in MODELS or not hasattr(MODELS[emotion_key], 'history') or MODELS[emotion_key].history.empty:
            add_log("WARNING", f"Model/histori untuk summary '{emotion_key}' tidak ada.")
            accuracies[emotion_key], trends[emotion_key] = 0, "N/A"
            continue
        model = MODELS[emotion_key]
        future = model.make_future_dataframe(periods=summary_forecast_days, freq=summary_freq)
        forecast = model.predict(future)
        accuracies[emotion_key], trends[emotion_key] = get_forecast_accuracy(model, forecast), get_weekly_trend(forecast) # Menggunakan tren mingguan
    add_log("INFO", "Berhasil memproses permintaan ke '/api/summary'.")
    return jsonify({"accuracies": accuracies, "trends": trends})

@app.route('/api/distribution', methods=['GET'])
def api_distribution():
    add_log("INFO", f"Menerima permintaan GET ke '/api/distribution' dengan args: {request.args}")
    time_range_param = request.args.get('range', 'today')
    end_date = datetime.now(timezone.utc) # Gunakan UTC untuk konsistensi
    if time_range_param == 'today': start_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
    elif time_range_param == 'week': start_date = (end_date - timedelta(days=6)).replace(hour=0, minute=0, second=0, microsecond=0)
    elif time_range_param == 'month': start_date = (end_date - timedelta(days=29)).replace(hour=0, minute=0, second=0, microsecond=0)
    else: 
        add_log("WARNING", f"Parameter 'range' tidak valid di '/api/distribution': {time_range_param}")
        return jsonify({"error": "Parameter 'range' tidak valid."}), 400

    distribution_result = []
    for emotion_name in AVAILABLE_EMOTIONS:
        if emotion_name not in MODELS or not hasattr(MODELS[emotion_name], 'history') or MODELS[emotion_name].history.empty:
            add_log("WARNING", f"Model/histori untuk distribusi '{emotion_name}' tidak ada.")
            distribution_result.append({"emotion": emotion_name, "count": 0}); continue
        history_df = MODELS[emotion_name].history.copy()
        
        # Ensure 'ds' is timezone-aware (UTC) for comparison with start/end_date which are aware
        history_df['ds'] = pd.to_datetime(history_df['ds'])
        if history_df['ds'].dt.tz is None: # Check if it's truly naive
            history_df['ds'] = history_df['ds'].dt.tz_localize('UTC', ambiguous='infer')

        filtered_history = history_df[(history_df['ds'] >= start_date) & (history_df['ds'] <= end_date)]
        emotion_count = 0
        if not filtered_history.empty and 'y' in filtered_history.columns:
            emotion_count = filtered_history['y'].sum()
            if pd.isna(emotion_count): emotion_count = 0
        distribution_result.append({"emotion": emotion_name, "count": int(emotion_count)})
    add_log("INFO", f"Berhasil memproses permintaan ke '/api/distribution' untuk range '{time_range_param}'.")
    return jsonify(distribution_result)

@app.route('/api/trends/today', methods=['GET'])
def api_trends_today():
    """Menghitung tren harian (hari ini vs kemarin) untuk setiap emosi."""
    add_log("INFO", "Menerima permintaan GET ke '/api/trends/today'.")
    daily_trends = {}
    
    # Define today and yesterday dates (UTC for consistency with Prophet data)
    today_dt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_dt = today_dt - timedelta(days=1)

    for emotion_key in AVAILABLE_EMOTIONS:
        if emotion_key not in MODELS:
            daily_trends[emotion_key] = "N/A (Model missing)"
            continue
        
        model = MODELS[emotion_key]
        if not hasattr(model, 'history') or model.history.empty:
            daily_trends[emotion_key] = "N/A (No history)"
            continue

        history_df = model.history.copy() 

        # Determine the last date in the model's history
        if not history_df.empty:
            last_history_date_naive = history_df['ds'].max() # Get the naive timestamp from Prophet
            
            if pd.isna(last_history_date_naive): # Handle NaN case
                 daily_trends[emotion_key] = "N/A (Last history date is NaN)"
                 continue

            # Convert to datetime, then always localize to UTC if it's naive, or convert if it has a different timezone
            last_history_date_tz_aware = pd.to_datetime(last_history_date_naive)
            if last_history_date_tz_aware.tzinfo is None:
                # This is the line where the error consistently happens.
                # Let's try a different approach: force assign timezone if naive.
                try:
                    last_history_date_tz_aware = last_history_date_tz_aware.tz_localize('UTC', ambiguous='infer')
                except ValueError:
                    # If 'infer' fails, explicitly specify how to handle ambiguous times,
                    # or assume no ambiguity for Prophet's daily 'ds' column.
                    # For daily data, 'infer' should typically work or 'False' can be used if no DST.
                    last_history_date_tz_aware = last_history_date_tz_aware.tz_localize('UTC', ambiguous=False)
            else:
                last_history_date_tz_aware = last_history_date_tz_aware.tz_convert('UTC') # Convert if already localized to something else

        else:
            daily_trends[emotion_key] = "N/A (No history data available for specific emotion)"
            continue

        # Calculate the number of days to predict to cover 'today_dt'
        periods_needed = 0
        if today_dt > last_history_date_tz_aware: # Compare timezone-aware dates
            periods_needed = (today_dt - last_history_date_tz_aware).days
        
        # Predict at least 1 day ahead if history is up-to-date,
        # or more if history is old. Add 1 so today_dt is not the last point.
        # Ensure periods_needed is at least 1 for making future dataframe
        future_df = model.make_future_dataframe(periods=max(1, periods_needed + 1), freq='D')
        forecast = model.predict(future_df)

        # Ensure 'ds' column is timezone-aware UTC for comparison
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        if forecast['ds'].dt.tz is None: # Check if it's truly naive
            forecast['ds'] = forecast['ds'].dt.tz_localize('UTC', ambiguous='infer')
        else:
            forecast['ds'] = forecast['ds'].dt.tz_convert('UTC') # Convert if already localized to something else

        today_forecast_series = forecast[forecast['ds'] == today_dt]['yhat']
        yesterday_forecast_series = forecast[forecast['ds'] == yesterday_dt]['yhat']

        if today_forecast_series.empty or yesterday_forecast_series.empty:
            add_log("WARNING", f"No forecast data for today or yesterday for emotion '{emotion_key}'.")
            daily_trends[emotion_key] = "N/A (Data missing)"
            continue

        today_yhat = today_forecast_series.iloc[0]
        yesterday_yhat = yesterday_forecast_series.iloc[0]

        if pd.isna(today_yhat) or pd.isna(yesterday_yhat):
            daily_trends[emotion_key] = "N/A (Prediksi NaN)"
            continue
        
        if yesterday_yhat == 0:
            trend_val = '+0.00%' if today_yhat >= 0 else '-0.00%'
        else:
            delta = ((today_yhat - yesterday_yhat) / yesterday_yhat) * 100
            trend_val = f"{'+' if delta >= 0 else ''}{round(delta, 2)}%"
        
        daily_trends[emotion_key] = trend_val

    add_log("INFO", "Berhasil memproses permintaan ke '/api/trends/today'.")
    return jsonify(daily_trends)

@app.route('/api/logs', methods=['GET'])
def get_api_logs():
    """Mengembalikan log API yang tersimpan di memori."""
    add_log("INFO", "Menerima permintaan GET ke '/api/logs'.")
    # Mengembalikan salinan agar tidak termodifikasi secara tidak sengaja
    return jsonify(list(API_LOGS))


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0') # host='0.0.0.0' agar bisa diakses dari luar kontainer jika perlu