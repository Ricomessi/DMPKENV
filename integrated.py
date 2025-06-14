from flask import Flask, jsonify, request
import mysql.connector
import pandas as pd
from flask_cors import CORS
import pickle
from datetime import datetime, timedelta, timezone
import logging
import math # Untuk math.isnan

# --- Flask Application Initialization ---
app = Flask(__name__)

# --- Configure Logging ---
# logging.basicConfig(level=logging.INFO) # Sudah di-cover oleh app.logger jika dikonfigurasi
app.logger.setLevel(logging.INFO) # Set Flask's logger level

# --- Global CORS Configuration ---
# Pastikan Next.js Anda berjalan di http://localhost:3000
origins = ["http://localhost:3000"] # Sesuaikan jika port Next.js Anda berbeda
CORS(app, supports_credentials=True, origins=origins)

# --- Database Configuration ---
DB_CONFIG = {
    'host': '34.128.100.191', # Pastikan IP ini bisa dijangkau oleh server Flask
    'user': 'root',
    'password': 'admin',
    'database': 'emotion_trendbox'
}

# --- Emotion Forecasting Model Configuration ---
MODELS = {}
AVAILABLE_EMOTIONS = ['angry', 'fear', 'happy', 'sad', 'surprised']
# Pastikan path 'models/' ini benar relatif terhadap direktori tempat app.py dijalankan
MODEL_PATH_TEMPLATE = 'models/model_prophet_{emotion}.pkl'

# --- In-memory API Logs for Forecasting API ---
FORECAST_API_LOGS = []

def add_forecast_log(level, message):
    """Adds a log entry to the FORECAST_API_LOGS list and logs via app.logger."""
    timestamp = datetime.now(timezone.utc).isoformat()
    log_entry = {"timestamp": timestamp, "level": level, "message": message}
    FORECAST_API_LOGS.append(log_entry)
    if len(FORECAST_API_LOGS) > 200: # Batasi jumlah log dalam memori
        FORECAST_API_LOGS.pop(0)
    
    # Gunakan app.logger Flask untuk output ke konsol/file
    if level.upper() == "ERROR":
        app.logger.error(f"[Forecast API Log] {message}")
    elif level.upper() == "WARNING":
        app.logger.warning(f"[Forecast API Log] {message}")
    else: # INFO atau level lainnya
        app.logger.info(f"[Forecast API Log] {message}")


add_forecast_log("INFO", "Combined Flask API starting up.")

# --- Load Forecasting Models ---
for emotion in AVAILABLE_EMOTIONS:
    try:
        with open(MODEL_PATH_TEMPLATE.format(emotion=emotion), 'rb') as f:
            MODELS[emotion] = pickle.load(f)
        add_forecast_log("INFO", f"Model for emotion '{emotion}' successfully loaded from {MODEL_PATH_TEMPLATE.format(emotion=emotion)}.")
    except FileNotFoundError:
        add_forecast_log("WARNING", f"Model file for emotion '{emotion}' not found at {MODEL_PATH_TEMPLATE.format(emotion=emotion)}. Please check the path.")
    except Exception as e:
        add_forecast_log("ERROR", f"Failed to load model for emotion '{emotion}'. Error: {e}")

# --- Database Connection Helper ---
def get_db_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        app.logger.info("Database connection successful.")
        return conn
    except mysql.connector.Error as err:
        app.logger.error(f"Error connecting to database: {err}. Check DB_CONFIG, network, and credentials.")
        raise # Re-raise the error agar bisa ditangani oleh endpoint

# --- Forecasting Helper Functions (from original forecasting app) ---
def get_forecast_accuracy(model, forecast_df_full):
    model_name_for_log = getattr(model, 'model_name', 'unknown_emotion') # Untuk logging
    if not hasattr(model, 'history') or model.history.empty:
        add_forecast_log("WARNING", f"Model '{model_name_for_log}' has no history for accuracy calculation.")
        return 0.0 # Kembalikan float agar konsisten

    # Pastikan kolom 'ds' dan 'y' ada di model.history
    if 'ds' not in model.history.columns or 'y' not in model.history.columns:
        add_forecast_log("WARNING", f"Model history for '{model_name_for_log}' is missing 'ds' or 'y' columns for accuracy.")
        return 0.0

    merged_df = pd.merge(forecast_df_full[['ds', 'yhat']], model.history[['ds', 'y']], on='ds', how='inner')
    
    if merged_df.empty:
        add_forecast_log("INFO", f"For model '{model_name_for_log}', no overlapping data between forecast and history for accuracy.")
        return 0.0
        
    # Hapus baris jika actual (y) atau prediksi history (yhat) adalah NaN
    merged_df.dropna(subset=['y', 'yhat'], inplace=True) 
    if merged_df.empty:
        add_forecast_log("INFO", f"For model '{model_name_for_log}', data for accuracy became empty after dropping NaNs.")
        return 0.0

    y_actual = merged_df['y'].astype(float) # Pastikan tipe data float
    y_hat_hist = merged_df['yhat'].astype(float) # Pastikan tipe data float

    if len(y_actual) == 0:
        return 0.0
    
    # Perhitungan MAPE yang lebih aman
    errors = abs(y_hat_hist - y_actual)
    non_zero_actuals_mask = (y_actual != 0)
    
    if non_zero_actuals_mask.sum() == 0: # Jika semua actuals adalah nol
        # Jika semua prediksi juga nol, akurasi 100%, jika tidak 0%
        return 100.0 if (y_hat_hist == 0).all() else 0.0

    # Hitung persentase error hanya untuk actuals yang tidak nol
    percentage_errors = errors[non_zero_actuals_mask] / y_actual[non_zero_actuals_mask]
    mape = percentage_errors.mean() * 100
    
    if pd.isna(mape): # Jika mape adalah NaN (misal, tidak ada non_zero_actuals)
        accuracy = 0.0
    else:
        accuracy = 100.0 - mape
    
    # Pastikan akurasi antara 0 dan 100
    accuracy = max(0.0, min(100.0, accuracy))
    return round(accuracy, 2)


def get_weekly_trend(forecast_df_full):
    if 'yhat' not in forecast_df_full.columns or len(forecast_df_full) < 14 :
        return 'N/A' # Tidak cukup data
    
    # Pastikan 'yhat' adalah numerik, ganti non-numerik dengan NaN
    forecast_df_full['yhat'] = pd.to_numeric(forecast_df_full['yhat'], errors='coerce')
    # Hapus baris dengan yhat NaN setelah konversi
    forecast_df_full.dropna(subset=['yhat'], inplace=True)

    if len(forecast_df_full) < 14: # Cek ulang setelah dropna
        return 'N/A'

    last_week_yhat = forecast_df_full.tail(7)['yhat']
    prev_week_yhat = forecast_df_full.tail(14).head(7)['yhat']

    if last_week_yhat.empty or prev_week_yhat.empty: # Jika salah satu kosong setelah filter
        return 'N/A'

    last_week_mean = last_week_yhat.mean()
    prev_week_mean = prev_week_yhat.mean()

    if pd.isna(last_week_mean) or pd.isna(prev_week_mean): # Jika mean adalah NaN
        return 'N/A'

    if prev_week_mean == 0:
        if last_week_mean == 0:
            return '+0.00%'
        # Jika prev_week_mean adalah 0 dan last_week_mean tidak, ini adalah perubahan tak terhingga.
        # Anda mungkin ingin mengembalikan 'Infinite Increase/Decrease' atau 'N/A'.
        # Untuk kesederhanaan, kita bisa anggap ini perubahan besar.
        return '+100.00%' if last_week_mean > 0 else '-100.00%' # Contoh representasi
    else:
        # Gunakan abs(prev_week_mean) di pembagi untuk persentase perubahan yang lebih stabil
        delta = ((last_week_mean - prev_week_mean) / abs(prev_week_mean)) * 100 
        return f"{'+' if delta >= 0 else ''}{round(delta, 2)}%"


# --- Combined API Endpoints ---

@app.route('/api/summary', methods=['GET', 'OPTIONS'])
def db_summary():
    if request.method == 'OPTIONS':
        app.logger.info("--- /api/summary (DB) OPTIONS request hit ---")
        return '', 200 
    
    conn = None
    try:
        app.logger.info("--- /api/summary (DB) GET request hit ---")
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT COUNT(*) AS count FROM emotion_tracking WHERE DATE(timestamp) = CURDATE()")
        detected_faces_row = cursor.fetchone()
        detected_faces = detected_faces_row['count'] if detected_faces_row else 0

        cursor.execute("""
            SELECT emotion FROM (
                SELECT emotion, COUNT(*) AS count FROM emotion_tracking
                WHERE DATE(timestamp) = CURDATE()
                GROUP BY emotion
                ORDER BY count DESC
                LIMIT 1
            ) AS t
        """)
        row = cursor.fetchone()
        dominant_emotion = row['emotion'] if row else 'N/A'

        cursor.execute("SELECT COUNT(*) AS total FROM emotion_tracking WHERE YEARWEEK(timestamp,1) = YEARWEEK(CURDATE(),1)")
        total_this_week_row = cursor.fetchone()
        total_this_week = total_this_week_row['total'] if total_this_week_row and total_this_week_row['total'] else 1

        cursor.execute("SELECT COUNT(*) AS total FROM emotion_tracking WHERE YEARWEEK(timestamp,1) = YEARWEEK(CURDATE() - INTERVAL 7 DAY,1)")
        total_last_week_row = cursor.fetchone()
        total_last_week = total_last_week_row['total'] if total_last_week_row and total_last_week_row['total'] else 1

        cursor.execute("SELECT emotion, COUNT(*) AS count FROM emotion_tracking WHERE YEARWEEK(timestamp,1) = YEARWEEK(CURDATE(),1) GROUP BY emotion")
        this_week_counts_raw = cursor.fetchall()
        this_week = {r['emotion']: (r['count'] / total_this_week) * 100 for r in this_week_counts_raw}


        cursor.execute("SELECT emotion, COUNT(*) AS count FROM emotion_tracking WHERE YEARWEEK(timestamp,1) = YEARWEEK(CURDATE() - INTERVAL 7 DAY,1) GROUP BY emotion")
        last_week_counts_raw = cursor.fetchall()
        last_week = {r['emotion']: (r['count'] / total_last_week) * 100 for r in last_week_counts_raw}


        changes = {}
        all_emotions_db = set(this_week.keys()).union(last_week.keys())
        if all_emotions_db : # Cek jika ada emosi
            for emotion_db in all_emotions_db:
                current = this_week.get(emotion_db, 0.0)
                last = last_week.get(emotion_db, 0.0)
                if last == 0:
                    changes[emotion_db] = 100.0 if current > 0 else 0.0 # Perubahan dari 0 ke X adalah 100%
                else:
                    changes[emotion_db] = round(((current - last) / last) * 100, 2)
        
        return jsonify({
            'detected_faces': detected_faces,
            'dominant_emotion': dominant_emotion,
            'weekly_changes': changes # Ini adalah persentase perubahan
        })
    except mysql.connector.Error as db_err:
        app.logger.error(f"Database error in /api/summary (DB): {db_err}", exc_info=True)
        return jsonify({"error": "Database error", "message": str(db_err)}), 500
    except Exception as e:
        app.logger.error(f"An unexpected error occurred in /api/summary (DB): {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred", "message": str(e)}), 500
    finally:
        if conn and conn.is_connected():
            conn.close()
            app.logger.info("--- Database connection closed for /api/summary (DB) ---")

@app.route('/api/logs', methods=['GET', 'OPTIONS'])
def db_logs():
    if request.method == 'OPTIONS':
        app.logger.info("--- /api/logs (DB) OPTIONS request hit ---")
        return '', 200

    conn = None
    try:
        app.logger.info("--- /api/logs (DB) GET request hit ---")
        conn = get_db_connection()
        
        query = """
            SELECT timestamp, user_id AS person, emotion, confidence
            FROM emotion_tracking
            ORDER BY timestamp DESC LIMIT 50
        """
        df = pd.read_sql(query, conn)
        app.logger.info(f"--- pd.read_sql for /api/logs (DB) executed, df has {len(df)} rows ---")
                
        if not df.empty:
            # Konversi timestamp ke string ISO format agar aman untuk JSON
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%dT%H:%M:%S')
            # Ganti NaN dengan None (null di JSON) jika ada
            df.fillna(value=pd.NA, inplace=True) # pd.NA lebih modern, tapi None juga oke
            records = df.to_dict(orient='records')
            # Iterasi untuk memastikan None jika pd.NA
            cleaned_records = []
            for record in records:
                cleaned_record = {k: (None if pd.isna(v) else v) for k, v in record.items()}
                cleaned_records.append(cleaned_record)
            return jsonify(cleaned_records)
        else:
            app.logger.info("--- /api/logs (DB) returning empty list as DataFrame is empty ---")
            return jsonify([]) 
                
    except mysql.connector.Error as db_err:
        app.logger.error(f"Database error in /api/logs (DB): {db_err}", exc_info=True)
        return jsonify({"error": "Database error", "message": str(db_err)}), 500
    except Exception as e:
        app.logger.error(f"An unexpected error occurred in /api/logs (DB): {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred", "message": str(e)}), 500
    finally:
        if conn and conn.is_connected():
            conn.close()
            app.logger.info("--- Database connection closed for /api/logs (DB) ---")

@app.route('/api/distribution', methods=['GET', 'OPTIONS'])
def db_distribution():
    if request.method == 'OPTIONS':
        app.logger.info("--- /api/distribution (DB) OPTIONS request hit ---")
        return '', 200

    conn = None
    try:
        range_param = request.args.get('range', 'today')
        app.logger.info(f"--- /api/distribution (DB) GET request hit with range: {range_param} ---")
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        condition = ""
        if range_param == 'today':
            condition = "DATE(timestamp) = CURDATE()"
        elif range_param == 'week':
            condition = "YEARWEEK(timestamp, 1) = YEARWEEK(CURDATE(), 1)"
        elif range_param == 'month':
            condition = "YEAR(timestamp) = YEAR(CURDATE()) AND MONTH(timestamp) = MONTH(CURDATE())"
        else:
            app.logger.warning(f"Invalid 'range' parameter for /api/distribution (DB): {range_param}. Defaulting to 'today'.")
            condition = "DATE(timestamp) = CURDATE()" 

        query = f"""
            SELECT emotion, COUNT(*) AS count FROM emotion_tracking
            WHERE {condition}
            GROUP BY emotion
        """
        cursor.execute(query)
        data = cursor.fetchall()
        return jsonify(data)
    except mysql.connector.Error as db_err:
        app.logger.error(f"Database error in /api/distribution (DB): {db_err}", exc_info=True)
        return jsonify({"error": "Database error", "message": str(db_err)}), 500
    except Exception as e:
        app.logger.error(f"An unexpected error occurred in /api/distribution (DB): {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred", "message": str(e)}), 500
    finally:
        if conn and conn.is_connected():
            conn.close()
            app.logger.info(f"--- Database connection closed for /api/distribution (DB) range: {range_param} ---")


@app.route('/api/trends/today', methods=['GET', 'OPTIONS'])
def db_trends_today():
    if request.method == 'OPTIONS':
        app.logger.info("--- /api/trends/today (DB) OPTIONS request hit ---")
        return '', 200

    conn = None
    try:
        app.logger.info("--- /api/trends/today (DB) GET request hit ---")
        conn = get_db_connection()
        df = pd.read_sql("""
            SELECT emotion, HOUR(timestamp) as hour, COUNT(*) as count
            FROM emotion_tracking
            WHERE DATE(timestamp) = CURDATE()
            GROUP BY emotion, hour
            ORDER BY hour
        """, conn)
        result = {}
        if not df.empty:
            for emotion_val, group in df.groupby('emotion'):
                result[emotion_val] = {'hours': group['hour'].tolist(), 'counts': group['count'].tolist()}
        return jsonify(result)
    except mysql.connector.Error as db_err:
        app.logger.error(f"Database error in /api/trends/today (DB): {db_err}", exc_info=True)
        return jsonify({"error": "Database error", "message": str(db_err)}), 500
    except Exception as e:
        app.logger.error(f"An unexpected error occurred in /api/trends/today (DB): {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred", "message": str(e)}), 500
    finally:
        if conn and conn.is_connected():
            conn.close()
            app.logger.info("--- Database connection closed for /api/trends/today (DB) ---")

@app.route('/api/trends/weekly', methods=['GET', 'OPTIONS'])
def db_trends_weekly():
    if request.method == 'OPTIONS':
        app.logger.info("--- /api/trends/weekly (DB) OPTIONS request hit ---")
        return '', 200

    conn = None
    try:
        app.logger.info("--- /api/trends/weekly (DB) GET request hit ---")
        conn = get_db_connection()
        query = """
            SELECT
                emotion,
                WEEKDAY(timestamp) as day_of_week, 
                COUNT(*) as count
            FROM emotion_tracking
            WHERE YEARWEEK(timestamp, 1) = YEARWEEK(CURDATE(), 1)
            GROUP BY emotion, day_of_week
            ORDER BY day_of_week
        """
        df = pd.read_sql(query, conn)
        result = {}
        if not df.empty:
            for emotion_val, group in df.groupby('emotion'):
                result[emotion_val] = {'days': group['day_of_week'].tolist(), 'counts': group['count'].tolist()}
        return jsonify(result)
    except mysql.connector.Error as db_err:
        app.logger.error(f"Database error in /api/trends/weekly (DB): {db_err}", exc_info=True)
        return jsonify({"error": "Database error", "message": str(db_err)}), 500
    except Exception as e:
        app.logger.error(f"An unexpected error occurred in /api/trends/weekly (DB): {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred", "message": str(e)}), 500
    finally:
        if conn and conn.is_connected():
            conn.close()
            app.logger.info("--- Database connection closed for /api/trends/weekly (DB) ---")


@app.route('/api/trends', methods=['GET', 'OPTIONS'])
def db_trends_all():
    if request.method == 'OPTIONS':
        app.logger.info("--- /api/trends (DB) OPTIONS request hit ---")
        return '', 200

    conn = None
    try:
        app.logger.info("--- /api/trends (DB) GET request hit ---")
        conn = get_db_connection()
        df = pd.read_sql("""
            SELECT emotion, DATE(timestamp) as date, COUNT(*) as count
            FROM emotion_tracking
            GROUP BY emotion, DATE(timestamp)
            ORDER BY date
        """, conn)
        result = {}
        if not df.empty:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d') # Format tanggal
            for emotion_val, group in df.groupby('emotion'):
                result[emotion_val] = {'dates': group['date'].tolist(), 'counts': group['count'].tolist()}
        return jsonify(result)
    except mysql.connector.Error as db_err:
        app.logger.error(f"Database error in /api/trends (DB): {db_err}", exc_info=True)
        return jsonify({"error": "Database error", "message": str(db_err)}), 500
    except Exception as e:
        app.logger.error(f"An unexpected error occurred in /api/trends (DB): {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred", "message": str(e)}), 500
    finally:
        if conn and conn.is_connected():
            conn.close()
            app.logger.info("--- Database connection closed for /api/trends (DB) ---")


# Endpoint from original Emotion Forecasting API
@app.route('/api/forecast', methods=['POST', 'OPTIONS'])
def get_forecast_data():
    app.logger.info(f"--- {request.method} /api/forecast ---")
    if request.method == 'OPTIONS':
        app.logger.info("Responding to OPTIONS request for /api/forecast")
        return '', 200 

    add_forecast_log("INFO", f"Received POST request to '/api/forecast' with payload: {request.json}")
    
    try:
        payload = request.get_json()
        if not payload: 
            add_forecast_log("WARNING", "Request to '/api/forecast' without JSON payload.")
            return jsonify({"error": "Invalid request payload"}), 400
            
        selected_emotion = payload.get('emotion', 'happy')
        if selected_emotion not in MODELS: 
            add_forecast_log("ERROR", f"Model for emotion '{selected_emotion}' not found during request to '/api/forecast'.")
            return jsonify({"error": f"Model '{selected_emotion}' not found"}), 404
            
        model = MODELS[selected_emotion]
        if not hasattr(model, 'history') or model.history.empty: 
            add_forecast_log("ERROR", f"Model '{selected_emotion}' has no history data during request to '/api/forecast'.")
            # Kembalikan data default jika tidak ada history, agar frontend tidak error
            return jsonify({
                'forecast_points': [], # List kosong untuk chart
                'accuracy': 0.0,
                'trend': 'N/A',
                'emotion': selected_emotion, 
                'message': f"Model '{selected_emotion}' has no history data."
            }), 200 # Gunakan 200 OK karena ini kondisi data, bukan error server

        forecast_days_count = int(payload.get('forecast_days', 7))
        granularity_frontend = payload.get('granularity', 'daily')
        prophet_freq = {"hourly": "H", "daily": "D", "weekly": "W", "monthly": "M"}.get(granularity_frontend.lower(), 'D')
        start_date_str, end_date_str = payload.get('start_date'), payload.get('end_date')

        future_df = model.make_future_dataframe(periods=forecast_days_count, freq=prophet_freq)
        forecast_df_full = model.predict(future_df)
        
        # Gabungkan dengan history. 'ds' di history dan forecast_df_full harusnya sudah datetime
        output_df = pd.merge(forecast_df_full[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                             model.history[['ds', 'y']], 
                             on='ds', 
                             how='left') # 'left' join untuk mempertahankan semua tanggal forecast

        # Filter berdasarkan tanggal jika ada
        if start_date_str and end_date_str:
            try:
                # Konversi ds ke naive datetime untuk perbandingan yang konsisten
                output_df['ds'] = pd.to_datetime(output_df['ds']).dt.tz_localize(None)
                start_dt_filter = pd.to_datetime(start_date_str).tz_localize(None)
                end_dt_filter = pd.to_datetime(end_date_str).tz_localize(None)
                output_df = output_df[(output_df['ds'] >= start_dt_filter) & (output_df['ds'] <= end_dt_filter)]
            except Exception as e: 
                add_forecast_log("ERROR", f"Error parsing or applying date filter in '/api/forecast': {e}")
                # Lanjutkan tanpa filter jika ada error tanggal

        recharts_data_list_clean = []
        if not output_df.empty:
            df_for_recharts = output_df.rename(columns={'y': 'actual', 'ds': 'name'})
            df_for_recharts['name'] = pd.to_datetime(df_for_recharts['name']).dt.strftime('%Y-%m-%d') # Format 'name'
            
            # Konversi DataFrame ke list of dictionaries
            recharts_data_list_raw = df_for_recharts.to_dict(orient='records')

            # **PERBAIKAN UTAMA: Ganti float NaN dan pd.NaT dengan None (JSON null)**
            for record in recharts_data_list_raw:
                cleaned_record = {}
                for key, value in record.items():
                    if (isinstance(value, float) and math.isnan(value)) or pd.isna(value): # pd.isna() menangani NaN, NaT
                        cleaned_record[key] = None 
                    else:
                        cleaned_record[key] = value
                recharts_data_list_clean.append(cleaned_record)
        
        current_accuracy = get_forecast_accuracy(model, forecast_df_full)
        current_trend = get_weekly_trend(forecast_df_full)
        
        add_forecast_log("INFO", f"Successfully processed request to '/api/forecast' for emotion '{selected_emotion}'. Returning {len(recharts_data_list_clean)} data points. Accuracy: {current_accuracy}, Trend: {current_trend}")
        
        # Logging data sampel untuk debugging
        if recharts_data_list_clean:
             app.logger.debug(f"Sample data point for {selected_emotion} (first): {recharts_data_list_clean[0]}")
        else:
             app.logger.debug(f"No data points to return for {selected_emotion} after processing.")


        return jsonify({
            'forecast_points': recharts_data_list_clean,
            'accuracy': current_accuracy,
            'trend': current_trend,
            'emotion': selected_emotion, 
            'forecast_days': forecast_days_count, 
            'granularity': granularity_frontend
        })
    except Exception as e:
        add_forecast_log("ERROR", f"Major unhandled error in /api/forecast: {e}")
        app.logger.error(f"Exception in /api/forecast: {e}", exc_info=True) # Tampilkan traceback lengkap
        return jsonify({"error": "Internal server error on forecast", "details": str(e)}), 500


@app.route('/api/forecast_summary', methods=['GET', 'OPTIONS'])
def api_forecast_summary():
    app.logger.info(f"--- {request.method} /api/forecast_summary ---")
    if request.method == 'OPTIONS':
        return '', 200 

    add_forecast_log("INFO", "Received GET request to '/api/forecast_summary'.")
    summary_forecast_days, summary_freq = 7, 'D' # default 7 hari, frekuensi harian
    accuracies, trends = {}, {}
    
    for emotion_key in AVAILABLE_EMOTIONS:
        try:
            if emotion_key not in MODELS or not hasattr(MODELS[emotion_key], 'history') or MODELS[emotion_key].history.empty:
                add_forecast_log("WARNING", f"Model/history for forecast summary (emotion: '{emotion_key}') is missing.")
                accuracies[emotion_key], trends[emotion_key] = 0.0, "N/A" # Default jika model/history tidak ada
                continue # Lanjut ke emosi berikutnya
            
            model = MODELS[emotion_key]
            future = model.make_future_dataframe(periods=summary_forecast_days, freq=summary_freq)
            forecast = model.predict(future)
            accuracies[emotion_key] = get_forecast_accuracy(model, forecast)
            trends[emotion_key] = get_weekly_trend(forecast)
        except Exception as e:
            add_forecast_log("ERROR", f"Error processing summary for emotion '{emotion_key}': {e}")
            app.logger.error(f"Exception in /api/forecast_summary for emotion {emotion_key}: {e}", exc_info=True)
            accuracies[emotion_key], trends[emotion_key] = 0.0, "Error" # Tandai error per emosi


    add_forecast_log("INFO", "Successfully processed request to '/api/forecast_summary'.")
    return jsonify({"accuracies": accuracies, "trends": trends})


@app.route('/api/forecast_distribution', methods=['GET', 'OPTIONS'])
def api_forecast_distribution():
    app.logger.info(f"--- {request.method} /api/forecast_distribution with args: {request.args} ---")
    if request.method == 'OPTIONS':
        return '', 200

    add_forecast_log("INFO", f"Received GET request to '/api/forecast_distribution' with args: {request.args}")
    time_range_param = request.args.get('range', 'today')
    
    # Tentukan start_date dan end_date berdasarkan time_range_param
    end_date_utc = datetime.now(timezone.utc) # Akhir rentang adalah sekarang (UTC)
    if time_range_param == 'today': 
        start_date_utc = end_date_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    elif time_range_param == 'week': 
        start_date_utc = (end_date_utc - timedelta(days=6)).replace(hour=0, minute=0, second=0, microsecond=0)
    elif time_range_param == 'month': 
        # Asumsi 'month' adalah 30 hari terakhir dari hari ini
        start_date_utc = (end_date_utc - timedelta(days=29)).replace(hour=0, minute=0, second=0, microsecond=0)
    else: 
        add_forecast_log("WARNING", f"Invalid 'range' parameter in '/api/forecast_distribution': {time_range_param}. Defaulting to 'today'.")
        return jsonify({"error": "Invalid 'range' parameter. Use 'today', 'week', or 'month'."}), 400

    distribution_result = []
    for emotion_name in AVAILABLE_EMOTIONS:
        try:
            if emotion_name not in MODELS or not hasattr(MODELS[emotion_name], 'history') or MODELS[emotion_name].history.empty:
                add_forecast_log("WARNING", f"Model/history for forecast distribution (emotion: '{emotion_name}') is missing.")
                distribution_result.append({"emotion": emotion_name, "count": 0}); 
                continue
            
            history_df = MODELS[emotion_name].history.copy()
            history_df['ds'] = pd.to_datetime(history_df['ds']) # Pastikan 'ds' adalah datetime

            # Pastikan 'ds' di history_df adalah timezone-aware (UTC) untuk perbandingan yang benar
            if history_df['ds'].dt.tz is None:
                history_df['ds'] = history_df['ds'].dt.tz_localize('UTC', ambiguous='infer') # Atau zona waktu data Anda jika diketahui
            else:
                history_df['ds'] = history_df['ds'].dt.tz_convert('UTC') # Konversi ke UTC jika sudah aware

            # Filter history_df berdasarkan rentang tanggal UTC
            filtered_history = history_df[(history_df['ds'] >= start_date_utc) & (history_df['ds'] <= end_date_utc)]
            
            emotion_count = 0
            if not filtered_history.empty and 'y' in filtered_history.columns:
                # Pastikan 'y' adalah numerik dan ganti NaN dengan 0 sebelum sum
                emotion_count = pd.to_numeric(filtered_history['y'], errors='coerce').fillna(0).sum()
            
            distribution_result.append({"emotion": emotion_name, "count": int(emotion_count)})
        except Exception as e:
            add_forecast_log("ERROR", f"Error processing distribution for emotion '{emotion_name}': {e}")
            app.logger.error(f"Exception in /api/forecast_distribution for {emotion_name}: {e}", exc_info=True)
            distribution_result.append({"emotion": emotion_name, "count": 0, "error": str(e)})

    add_forecast_log("INFO", f"Successfully processed request to '/api/forecast_distribution' for range '{time_range_param}'.")
    return jsonify(distribution_result)


@app.route('/api/forecast_trends_today', methods=['GET', 'OPTIONS'])
def api_forecast_trends_today():
    app.logger.info(f"--- {request.method} /api/forecast_trends_today ---")
    if request.method == 'OPTIONS':
        return '', 200

    add_forecast_log("INFO", "Received GET request to '/api/forecast_trends_today'.")
    daily_trends = {}
    
    # Tentukan 'today' dan 'yesterday' dalam UTC untuk konsistensi dengan data model (jika model dilatih dengan UTC)
    # Jika model Anda menggunakan waktu lokal, sesuaikan ini.
    today_dt_utc = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_dt_utc = today_dt_utc - timedelta(days=1)

    for emotion_key in AVAILABLE_EMOTIONS:
        try:
            if emotion_key not in MODELS:
                daily_trends[emotion_key] = "N/A (Model missing)"
                continue
            
            model = MODELS[emotion_key]
            if not hasattr(model, 'history') or model.history.empty:
                daily_trends[emotion_key] = "N/A (No history)"
                continue

            # Dapatkan tanggal terakhir dari history untuk menentukan berapa banyak periode forecast yang dibutuhkan
            last_history_date_naive = pd.to_datetime(model.history['ds'].max())
            if pd.isna(last_history_date_naive):
                 daily_trends[emotion_key] = "N/A (History date invalid)"
                 continue

            # Asumsikan history 'ds' adalah naive atau bisa dikonversi ke UTC
            last_history_date_utc = last_history_date_naive.tz_localize('UTC', ambiguous='infer') if last_history_date_naive.tzinfo is None else last_history_date_naive.tz_convert('UTC')

            # Hitung periode yang dibutuhkan untuk forecast hingga hari ini
            periods_needed = 0
            if today_dt_utc > last_history_date_utc:
                periods_needed = (today_dt_utc - last_history_date_utc).days
            
            # Forecast setidaknya untuk 1 hari ke depan, atau lebih jika history tertinggal
            future_df = model.make_future_dataframe(periods=max(1, periods_needed + 1), freq='D') 
            forecast = model.predict(future_df)

            # Pastikan 'ds' di forecast adalah UTC untuk perbandingan
            forecast['ds'] = pd.to_datetime(forecast['ds'])
            if forecast['ds'].dt.tz is None:
                forecast['ds'] = forecast['ds'].dt.tz_localize('UTC', ambiguous='infer')
            else:
                forecast['ds'] = forecast['ds'].dt.tz_convert('UTC')

            # Dapatkan prediksi untuk hari ini dan kemarin
            today_forecast_series = forecast[forecast['ds'] == today_dt_utc]['yhat']
            yesterday_forecast_series = forecast[forecast['ds'] == yesterday_dt_utc]['yhat']

            if today_forecast_series.empty or yesterday_forecast_series.empty:
                add_forecast_log("WARNING", f"No forecast data for today or yesterday for emotion '{emotion_key}'.")
                daily_trends[emotion_key] = "N/A (Data missing for trend)"
                continue

            today_yhat = today_forecast_series.iloc[0]
            yesterday_yhat = yesterday_forecast_series.iloc[0]

            if pd.isna(today_yhat) or pd.isna(yesterday_yhat):
                daily_trends[emotion_key] = "N/A (Prediction NaN for trend)"
                continue
            
            if yesterday_yhat == 0:
                trend_val = '+0.00%' if today_yhat >= 0 else '-0.00%' # Atau representasi lain untuk perubahan dari nol
            else:
                delta = ((today_yhat - yesterday_yhat) / abs(yesterday_yhat)) * 100 # Gunakan abs untuk penyebut
                trend_val = f"{'+' if delta >= 0 else ''}{round(delta, 2)}%"
            
            daily_trends[emotion_key] = trend_val
        except Exception as e:
            add_forecast_log("ERROR", f"Error processing daily trend for emotion '{emotion_key}': {e}")
            app.logger.error(f"Exception in /api/forecast_trends_today for {emotion_key}: {e}", exc_info=True)
            daily_trends[emotion_key] = "Error"


    add_forecast_log("INFO", "Successfully processed request to '/api/forecast_trends_today'.")
    return jsonify(daily_trends)


@app.route('/api/forecast_logs', methods=['GET', 'OPTIONS'])
def get_forecast_api_logs():
    app.logger.info(f"--- {request.method} /api/forecast_logs ---")
    if request.method == 'OPTIONS':
        return '', 200
    add_forecast_log("INFO", "Received GET request to '/api/forecast_logs'.")
    return jsonify(list(FORECAST_API_LOGS)) # Kirim salinan list


if __name__ == '__main__':
    # Add a simple test to ensure DB connection at startup (optional but good for dev)
    print("Attempting initial DB connection test...")
    try:
        conn = get_db_connection()
        if conn:
            print("Initial DB connection successful!")
            conn.close()
        else:
            # Ini seharusnya tidak terjadi jika get_db_connection() raise error atau return conn
            print("Initial DB connection failed but no exception raised (conn is None). Check get_db_connection logic.")
    except Exception as e:
        print(f"Initial DB connection failed with error: {e}")

    # host='0.0.0.0' agar bisa diakses dari jaringan lokal (misal, dari Next.js yang berjalan di WSL atau kontainer lain)
    # debug=True akan mengaktifkan auto-reloader dan debugger Flask. Matikan di produksi.
    app.run(debug=True, port=5000, host='0.0.0.0')