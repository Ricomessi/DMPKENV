from flask import Flask, jsonify, request
import mysql.connector
import pandas as pd
from flask_cors import CORS
# import logging # Anda bisa menggunakan app.logger bawaan Flask

app = Flask(__name__)

# --- KONFIGURASI CORS GLOBAL ---
CORS(app, supports_credentials=True) # Izinkan semua, baik untuk pengembangan
# -------------------------------

DB_CONFIG = {
    'host': '34.128.100.191',
    'user': 'root',
    'password': 'admin',
    'database': 'emotion_trendbox'
}

def get_db_connection():
    # Tambahkan try-except di sini jika ingin penanganan error koneksi yang lebih baik
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        app.logger.error(f"Error connecting to database: {err}")
        raise # Re-raise error agar bisa ditangkap di endpoint jika perlu

@app.route('/api/summary')
def summary():
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Detected faces today
        cursor.execute("SELECT COUNT(*) AS count FROM emotion_tracking WHERE DATE(timestamp) = CURDATE()")
        detected_faces_row = cursor.fetchone()
        detected_faces = detected_faces_row['count'] if detected_faces_row else 0

        # Dominant emotion today
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

        # Weekly emotion change
        cursor.execute("SELECT COUNT(*) AS total FROM emotion_tracking WHERE YEARWEEK(timestamp,1) = YEARWEEK(CURDATE(),1)")
        total_this_week_row = cursor.fetchone()
        total_this_week = total_this_week_row['total'] if total_this_week_row and total_this_week_row['total'] else 1

        cursor.execute("SELECT COUNT(*) AS total FROM emotion_tracking WHERE YEARWEEK(timestamp,1) = YEARWEEK(CURDATE() - INTERVAL 7 DAY,1)")
        total_last_week_row = cursor.fetchone()
        total_last_week = total_last_week_row['total'] if total_last_week_row and total_last_week_row['total'] else 1

        cursor.execute("SELECT emotion, COUNT(*) AS count FROM emotion_tracking WHERE YEARWEEK(timestamp,1) = YEARWEEK(CURDATE(),1) GROUP BY emotion")
        this_week_counts = cursor.fetchall()
        this_week = {r['emotion']: r['count']/total_this_week for r in this_week_counts}

        cursor.execute("SELECT emotion, COUNT(*) AS count FROM emotion_tracking WHERE YEARWEEK(timestamp,1) = YEARWEEK(CURDATE() - INTERVAL 7 DAY,1) GROUP BY emotion")
        last_week_counts = cursor.fetchall()
        last_week = {r['emotion']: r['count']/total_last_week for r in last_week_counts}

        changes = {}
        all_emotions = set(this_week.keys()).union(last_week.keys())
        if all_emotions and (this_week_counts or last_week_counts): # Pemeriksaan yang lebih baik
            for emotion in all_emotions:
                current = this_week.get(emotion, 0)
                last = last_week.get(emotion, 0)
                if current == 0 and last == 0:
                    changes[emotion] = 0.0
                elif last == 0:
                    changes[emotion] = 100.0 if current > 0 else 0.0
                else:
                    changes[emotion] = round(((current - last) / last) * 100, 2)
        
        return jsonify({
            'detected_faces': detected_faces,
            'dominant_emotion': dominant_emotion,
            'weekly_changes': changes
        })
    except mysql.connector.Error as db_err:
        app.logger.error(f"Database error in /api/summary: {db_err}")
        return jsonify({"error": "Database error", "message": str(db_err)}), 500
    except Exception as e:
        app.logger.error(f"An unexpected error occurred in /api/summary: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred", "message": str(e)}), 500
    finally:
        if conn and conn.is_connected():
            conn.close()


@app.route('/api/logs')
def logs():
    conn = None
    try:
        app.logger.info("--- /api/logs endpoint hit ---")
        conn = get_db_connection()
        app.logger.info("--- Database connection successful for /api/logs ---")
        
        # PERUBAHAN: Menghapus 'source' dari query SQL
        query = """
            SELECT timestamp, user_id AS person, emotion, confidence
            FROM emotion_tracking
            ORDER BY timestamp DESC LIMIT 50
        """
        df = pd.read_sql(query, conn)
        app.logger.info(f"--- pd.read_sql for /api/logs executed, df has {len(df)} rows ---")
           
        if not df.empty:
            df['timestamp'] = df['timestamp'].astype(str)
        else:
            app.logger.info("--- /api/logs returning empty list as DataFrame is empty ---")
            return jsonify([]) 
               
        app.logger.info("--- Data processing for /api/logs successful, returning JSON ---")
        return jsonify(df.to_dict(orient='records'))
           
    except mysql.connector.Error as db_err:
        app.logger.error(f"Database error in /api/logs: {db_err}")
        return jsonify({"error": "Database error", "message": str(db_err)}), 500
    except pd.errors.DatabaseError as pd_db_err: # Error spesifik pandas untuk query
        app.logger.error(f"Pandas Database query error in /api/logs: {pd_db_err}")
        return jsonify({"error": "Database query error via pandas", "message": str(pd_db_err)}), 500
    except Exception as e:
        app.logger.error(f"An unexpected error occurred in /api/logs: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred", "message": str(e)}), 500
    finally:
        if conn and conn.is_connected():
            conn.close()
            app.logger.info("--- Database connection closed for /api/logs ---")


@app.route('/api/distribution')
def distribution():
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        range_param = request.args.get('range', 'today')

        if range_param == 'today':
            condition = "DATE(timestamp) = CURDATE()"
        elif range_param == 'week':
            condition = "YEARWEEK(timestamp, 1) = YEARWEEK(CURDATE(), 1)"
        elif range_param == 'month':
            condition = "YEAR(timestamp) = YEAR(CURDATE()) AND MONTH(timestamp) = MONTH(CURDATE())"
        else: # Fallback default
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
        app.logger.error(f"Database error in /api/distribution: {db_err}")
        return jsonify({"error": "Database error", "message": str(db_err)}), 500
    except Exception as e:
        app.logger.error(f"An unexpected error occurred in /api/distribution: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred", "message": str(e)}), 500
    finally:
        if conn and conn.is_connected():
            conn.close()


@app.route('/api/trends/today')
def trends_today():
    conn = None
    try:
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
            for emotion, group in df.groupby('emotion'):
                result[emotion] = {'hours': group['hour'].tolist(), 'counts': group['count'].tolist()}
        return jsonify(result)
    except mysql.connector.Error as db_err:
        app.logger.error(f"Database error in /api/trends/today: {db_err}")
        return jsonify({"error": "Database error", "message": str(db_err)}), 500
    except Exception as e:
        app.logger.error(f"An unexpected error occurred in /api/trends/today: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred", "message": str(e)}), 500
    finally:
        if conn and conn.is_connected():
            conn.close()


@app.route('/api/trends/weekly')
def trends_weekly():
    conn = None
    try:
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
            for emotion, group in df.groupby('emotion'):
                result[emotion] = {'days': group['day_of_week'].tolist(), 'counts': group['count'].tolist()}
        return jsonify(result)
    except mysql.connector.Error as db_err:
        app.logger.error(f"Database error in /api/trends/weekly: {db_err}")
        return jsonify({"error": "Database error", "message": str(db_err)}), 500
    except Exception as e:
        app.logger.error(f"An unexpected error occurred in /api/trends/weekly: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred", "message": str(e)}), 500
    finally:
        if conn and conn.is_connected():
            conn.close()


@app.route('/api/trends')
def trends():
    conn = None
    try:
        conn = get_db_connection()
        df = pd.read_sql("""
            SELECT emotion, DATE(timestamp) as date, COUNT(*) as count
            FROM emotion_tracking
            GROUP BY emotion, DATE(timestamp)
            ORDER BY date
        """, conn)
        result = {}
        if not df.empty:
            df['date'] = df['date'].astype(str)
            for emotion, group in df.groupby('emotion'):
                result[emotion] = {'dates': group['date'].tolist(), 'counts': group['count'].tolist()}
        return jsonify(result)
    except mysql.connector.Error as db_err:
        app.logger.error(f"Database error in /api/trends: {db_err}")
        return jsonify({"error": "Database error", "message": str(db_err)}), 500
    except Exception as e:
        app.logger.error(f"An unexpected error occurred in /api/trends: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred", "message": str(e)}), 500
    finally:
        if conn and conn.is_connected():
            conn.close()


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0') # Menambahkan host='0.0.0.0' jika perlu