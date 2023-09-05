from flask import Flask, render_template, request
import pandas as pd  # Import pandas for creating user data DataFrame
import joblib

rf_model = joblib.load("fighter_prediction_model.pkl")


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input_kd = float(request.form["kd"])
        user_input_opponent_kd = float(request.form["opponent_kd"])
        user_input_sig_str = float(request.form["user_input_sig_str"])
        user_input_opponent_sig_str = float(request.form["user_input_opponent_sig_str"])
        user_input_td = float(request.form["user_input_td"])
        user_input_opponent_td = float(request.form["user_input_opponent_td"])
        user_input_sub_att = float(request.form["user_input_sub_att"])
        user_input_opponent_sub_att = float(request.form["user_input_opponent_sub_att"])
        user_input_rev = float(request.form["user_input_rev"])
        user_input_opponent_rev = float(request.form["user_input_opponent_rev"])

        user_data = pd.DataFrame({
            'KD': [user_input_kd],
            'Opponent KD': [user_input_opponent_kd],
            'Sig. str. %': [user_input_sig_str],
            'Opponent Sig. str. %': [user_input_opponent_sig_str],
            'Td %': [user_input_td],
            'Opponent Td %': [user_input_opponent_td],
            'Sub. att': [user_input_sub_att],
            'Opponent Sub. att': [user_input_opponent_sub_att],
            'Rev.': [user_input_rev],
            'Opponent Rev.': [user_input_opponent_rev],
        })

        prediction = rf_model.predict(user_data)
        result = "Win" if prediction[0] == 1 else "Loss"
        return render_template("index.html", result=result)

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
