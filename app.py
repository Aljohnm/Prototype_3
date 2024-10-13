import os
import time
import logging
from datetime import datetime
from flask import (
    Flask,
    request,
    render_template,
    redirect,
    request,
    jsonify,
    session,
    send_from_directory,
)
from werkzeug.utils import secure_filename
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from bson.objectid import ObjectId
from urllib.parse import urlparse
import re
from sklearn.impute import SimpleImputer

# Set matplotlib to use the 'Agg' backend
import matplotlib

matplotlib.use("Agg")

app = Flask(__name__)
app.secret_key = "your_secret_key"

# MongoDB connection
client = MongoClient(
    "mongodb+srv://ravi:Qwert123@maincluster.ff8x9.mongodb.net/myproject_db?retryWrites=true&w=majority"
)
db = client.myproject_db
users_collection = db.users
user_questions = db.user_questions
url_checks = db.url_checks  # Collection for storing URL and result

# File Upload Settings
UPLOAD_FOLDER = "uploads/"
ALLOWED_EXTENSIONS = {"csv"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Clean dataset column names function
def clean_column_names(df):
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
    df.columns = df.columns.str.lower()  # Convert all column names to lowercase
    return df


# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Helper function to check file extension
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Helper function to manage the last 3 datasets
def manage_last_three_datasets():
    files = os.listdir(app.config["UPLOAD_FOLDER"])
    csv_files = [
        os.path.join(app.config["UPLOAD_FOLDER"], f)
        for f in files
        if f.endswith(".csv")
    ]
    csv_files.sort(key=os.path.getctime)  # Sort by creation time

    # If there are more than 3 files, delete the oldest ones
    if len(csv_files) > 3:
        for file_to_remove in csv_files[:-3]:
            os.remove(file_to_remove)


### Routes ###
# Route for the login page
@app.route("/")
def login():
    return render_template("login.html")


# Route to handle login
@app.route("/login", methods=["POST"])
def login_post():
    username = request.form["username"]
    password = request.form["password"]
    role = request.form["role"]

    # Find the user in MongoDB
    user = users_collection.find_one(
        {"username": username, "password": password, "role": role}
    )

    if user:
        session["username"] = username
        session["role"] = role

        # Redirect to the appropriate dashboard based on role
        if role == "admin":
            return redirect("/admin_dashboard")
        else:
            return redirect("/user_dashboard")
    else:
        return "Invalid username, password, or role!"


# Route for the admin dashboard
@app.route("/admin_dashboard")
def admin_dashboard():
    if "role" in session and session["role"] == "admin":
        return render_template("admin_dashboard.html")
    return redirect("/")


# Route for user dashboard (basic placeholder for now)
@app.route("/user_dashboard")
def user_dashboard():
    if "role" in session and session["role"] == "user":
        # Retrieve the latest 5 URLs and their results from MongoDB
        recent_urls = list(url_checks.find().sort("timestamp", -1).limit(5))
        return render_template("user_dashboard.html", recent_urls=recent_urls)
    return redirect("/")


# New route for the separate Upload Dataset page
@app.route("/upload_dataset", methods=["GET", "POST"])
def upload_dataset():
    if request.method == "GET":
        return render_template(
            "upload_dataset.html"
        )  # This is the new page to upload datasets

    elif request.method == "POST":
        if "dataset" not in request.files:
            return "No file part", 400

        file = request.files["dataset"]

        if file.filename == "":
            return "No selected file", 400

        if file:
            # Use a timestamp to make each file unique
            timestamp = int(time.time())
            filename = f"{timestamp}_{secure_filename(file.filename)}"
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

            # Manage the last 3 datasets for prediction
            manage_last_three_datasets()

            return "Dataset uploaded successfully!", 200

    return "File upload failed", 400


# Hybrid Model: Stacked Ensemble Learning
def hybrid_model(df):
    # Define feature columns and target variable
    features = [
        "memory_psstotal",
        "memory_privatedirty",
        "network_totaltransmittedbytes",
        "network_totalreceivedbytes",
        "battery_wakelock",
        "battery_service",
    ]
    target = "category"

    X = df[features]
    y = df[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define base learners
    estimators = [
        ("rf", RandomForestClassifier(n_estimators=100)),
        ("gb", GradientBoostingClassifier(n_estimators=100)),
    ]

    # Define Stacking Classifier
    clf = StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression(), cv=5
    )

    # Train the model
    clf.fit(X_train, y_train)

    # Predict on test data
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, clf


from sklearn.impute import SimpleImputer


# Prediction route with hybrid machine learning integration
@app.route("/data_prediction")
def data_prediction():
    if "role" in session and session["role"] == "admin":
        # Get the last 3 CSV datasets
        files = sorted(
            [f for f in os.listdir(app.config["UPLOAD_FOLDER"]) if f.endswith(".csv")],
            reverse=True,
        )[:3]
        if len(files) == 0:
            return "<h2>No datasets uploaded yet!</h2>"

        dfs = []
        try:
            # Load and concatenate the last 3 datasets
            for file in files:
                df = pd.read_csv(
                    os.path.join(app.config["UPLOAD_FOLDER"], file), on_bad_lines="skip"
                )
                df = clean_column_names(df)
                dfs.append(df)

            full_data = pd.concat(dfs)

            # Define dynamic thresholds for high memory usage (90th percentile)
            high_memory_threshold = full_data["memory_psstotal"].quantile(0.9)

            # High memory instances
            high_memory_instances = (
                full_data[full_data["memory_psstotal"] > high_memory_threshold]
                .groupby("category")
                .size()
            )
            total_high_memory_instances = high_memory_instances.sum()

            # Most network activity
            try:
                most_network_activity_category = (
                    full_data.groupby("category")["network_totaltransmittedbytes"]
                    .sum()
                    .idxmax()
                )
                most_network_bytes = (
                    full_data.groupby("category")["network_totaltransmittedbytes"]
                    .sum()
                    .max()
                )
            except ValueError:
                most_network_activity_category = "No data"
                most_network_bytes = 0

            # Most battery usage
            try:
                most_battery_usage_category = (
                    full_data.groupby("category")["battery_wakelock"].sum().idxmax()
                )
            except ValueError:
                most_battery_usage_category = "No data"

            # Rolling average for memory and network usage (using window of 3 datasets)
            rolling_memory_avg = full_data["memory_psstotal"].rolling(window=3).mean()
            rolling_network_avg = (
                full_data["network_totaltransmittedbytes"].rolling(window=3).mean()
            )

            # Compute risk score based on weighted memory, network, and battery usage
            full_data["risk_score"] = (
                (full_data["memory_psstotal"] / full_data["memory_psstotal"].max())
                * 0.4
                + (
                    full_data["network_totaltransmittedbytes"]
                    / full_data["network_totaltransmittedbytes"].max()
                )
                * 0.4
                + (full_data["battery_wakelock"] / full_data["battery_wakelock"].max())
                * 0.2
            )

            # Assign risk levels based on the risk score
            conditions = [
                (full_data["risk_score"] > 0.8),
                (full_data["risk_score"] > 0.5) & (full_data["risk_score"] <= 0.8),
                (full_data["risk_score"] <= 0.5),
            ]
            risk_labels = ["High", "Medium", "Low"]
            full_data["predicted_risk_category"] = np.select(
                conditions, risk_labels, default="Low"
            )

            # Hybrid Model Prediction (classification on the 'category' column)
            features = [
                "memory_psstotal",
                "memory_privatedirty",
                "network_totaltransmittedbytes",
                "network_totalreceivedbytes",
                "battery_wakelock",
                "battery_service",
            ]

            # Ensure all necessary features exist in the dataset
            missing_columns = [col for col in features if col not in full_data.columns]
            if missing_columns:
                return f"<h2>Missing required columns: {missing_columns}. Available columns: {list(full_data.columns)}</h2>"

            X = full_data[features]
            y = full_data["category"]

            # Handle missing values by imputing them with the mean
            imputer = SimpleImputer(strategy="mean")
            X = imputer.fit_transform(X)

            # Split the data for training and testing
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Hybrid model using stacking (RandomForest and GradientBoosting)
            estimators = [
                ("rf", RandomForestClassifier(n_estimators=100)),
                ("gb", GradientBoostingClassifier(n_estimators=100)),
            ]

            clf = StackingClassifier(
                estimators=estimators, final_estimator=LogisticRegression(), cv=5
            )

            # Train the hybrid model
            clf.fit(X_train, y_train)

            # Make predictions
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Predicted API calls over time (based on memory usage and network bytes)
            api_calls_over_time = (
                full_data.groupby(["category", "memory_psstotal"])
                .size()
                .unstack(fill_value=0)
                .sum(axis=1)
            )

            # Visualization 1: High Memory Instances by Category
            plt.figure(figsize=(10, 6))
            high_memory_instances.plot(kind="bar", color="skyblue")
            plt.title("High Memory Instances by Trojan Category")
            plt.ylabel("Number of High Memory Instances")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            memory_pred_path = os.path.join(
                app.config["UPLOAD_FOLDER"], "high_memory_pred.png"
            )
            plt.savefig(memory_pred_path)
            plt.close()

            # Visualization 2: Predicted API Calls Over Time
            plt.figure(figsize=(10, 6))
            api_calls_over_time.plot(kind="line", color=["red", "blue"], marker="o")
            plt.title("Predicted API Calls Over Time")
            plt.ylabel("Number of API Calls")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            api_pred_path = os.path.join(
                app.config["UPLOAD_FOLDER"], "api_calls_pred.png"
            )
            plt.savefig(api_pred_path)
            plt.close()

            # Prediction summary and detailed table
            prediction_summary = {
                "total_high_memory_instances": total_high_memory_instances,
                "highest_risk_category": (
                    high_memory_instances.idxmax()
                    if not high_memory_instances.empty
                    else "No data"
                ),
                "avg_risk_score": full_data["risk_score"].mean(),
                "predicted_risk_category": full_data["predicted_risk_category"].mode()[
                    0
                ],
                "most_network_activity": (
                    f"{most_network_activity_category} ({most_network_bytes} bytes)"
                    if most_network_activity_category != "No data"
                    else "No data"
                ),
                "most_battery_usage": (
                    most_battery_usage_category
                    if most_battery_usage_category != "No data"
                    else "No data"
                ),
                "critical_date": datetime.now().strftime("%Y-%m-%d"),
                "suggested_action": f'Focus security resources on monitoring high memory usage in {high_memory_instances.idxmax() if not high_memory_instances.empty else "the dataset"} and network activity in {most_network_activity_category}.',
            }

            # Create detailed results table
            detailed_results = full_data.groupby("category").agg(
                {
                    "memory_psstotal": "sum",
                    "memory_privatedirty": "sum",
                    "network_totaltransmittedbytes": "sum",
                    "network_totalreceivedbytes": "sum",
                    "battery_wakelock": "sum",
                    "battery_service": "sum",
                    "api_command_java.lang.runtime_exec": "sum",
                    "api_webview_android.webkit.webview_loadurl": "sum",
                }
            )

            # Render results in HTML
            return render_template(
                "data_prediction.html",
                prediction_summary=prediction_summary,
                detailed_results=detailed_results.to_html(),
                memory_pred_img=memory_pred_path,
                api_pred_img=api_pred_path,
            )

        except Exception as e:
            return f"<h2>Error processing datasets: {e}</h2>"

    return redirect("/")


@app.route("/data_visualization", methods=["GET"])
def data_visualization():
    if "role" in session and session["role"] == "admin":
        # Get the dataset and categories
        files = os.listdir(app.config["UPLOAD_FOLDER"])
        latest_file = max(
            [
                os.path.join(app.config["UPLOAD_FOLDER"], f)
                for f in files
                if f.endswith(".csv")
            ],
            key=os.path.getctime,
        )
        df = pd.read_csv(latest_file)
        df = clean_column_names(df)  # Clean the column names

        # Required columns for visualization
        required_columns = [
            "category",
            "memory_psstotal",
            "memory_privatedirty",
            "memory_heapsize",
            "api_command_java.lang.runtime_exec",
            "api_webview_android.webkit.webview_loadurl",
            "network_totalreceivedbytes",
            "network_totaltransmittedbytes",
            "battery_wakelock",
            "battery_service",
        ]

        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        available_columns = [col for col in required_columns if col in df.columns]

        # If columns are missing, inform the admin but still show other available visualizations
        if missing_columns:
            missing_message = f"Warning: The following columns are missing and could not be visualized: {', '.join(missing_columns)}"
        else:
            missing_message = None

        # Get unique categories for the dropdown
        categories = df["category"].unique()

        # Retrieve filter category from the URL query
        category = request.args.get(
            "category", "all"
        )  # Default to 'all' if not provided

        # Filter by category if a specific category is selected
        if category != "all":
            filtered_df = df[df["category"] == category]
        else:
            filtered_df = df

        # ------------- Visualization 1: Distribution of Trojan Categories -------------
        plt.figure(figsize=(8, 6))
        category_counts = df["category"].value_counts()
        colors = [
            "lightgray" if cat != category else "skyblue"
            for cat in category_counts.index
        ]
        category_counts.plot(kind="pie", autopct="%1.1f%%", colors=colors)
        plt.title(f"Distribution of Trojan Categories (Filtered by {category})")
        plt.ylabel("")
        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(left=0.15, bottom=0.2)
        category_pie_path = os.path.join(
            app.config["UPLOAD_FOLDER"], "category_pie_filtered.png"
        )
        plt.savefig(category_pie_path)
        plt.close()

        # ------------- Visualization 2: Memory Usage by Trojan Category -------------
        if "memory_psstotal" in available_columns:
            plt.figure(figsize=(10, 6))
            filtered_df.groupby("category").agg(
                {
                    "memory_psstotal": "mean",
                    "memory_privatedirty": "mean",
                    "memory_heapsize": "mean",
                }
            ).plot(kind="bar")
            plt.title(
                f"Memory Usage by Classified Trojan Category (Filtered by {category})"
            )
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.ylabel("Memory Usage (KB)")
            memory_usage_path = os.path.join(
                app.config["UPLOAD_FOLDER"], "memory_usage_filtered.png"
            )
            plt.savefig(memory_usage_path)
            plt.close()
        else:
            memory_usage_path = None

        # ------------- Visualization 3: Battery Usage by Trojan Category -------------
        if (
            "battery_wakelock" in available_columns
            and "battery_service" in available_columns
        ):
            plt.figure(figsize=(10, 6))
            filtered_df.groupby("category").agg(
                {"battery_wakelock": "sum", "battery_service": "sum"}
            ).plot(kind="bar", color=["yellow", "blue"])
            plt.title(f"Battery Usage by Trojan Category (Filtered by {category})")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.ylabel("Battery Usage (Counts)")
            battery_usage_path = os.path.join(
                app.config["UPLOAD_FOLDER"], "battery_usage_filtered.png"
            )
            plt.savefig(battery_usage_path)
            plt.close()
        else:
            battery_usage_path = None

        # ------------- Visualization 4: Network Usage by Trojan Category -------------
        if (
            "network_totalreceivedbytes" in available_columns
            and "network_totaltransmittedbytes" in available_columns
        ):
            plt.figure(figsize=(10, 6))
            filtered_df.groupby("category").agg(
                {
                    "network_totalreceivedbytes": "sum",
                    "network_totaltransmittedbytes": "sum",
                }
            ).plot(kind="bar", color=["red", "cyan"])
            plt.title(f"Network Usage by Trojan Category (Filtered by {category})")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.ylabel("Network Usage (Bytes)")
            network_usage_path = os.path.join(
                app.config["UPLOAD_FOLDER"], "network_usage_filtered.png"
            )
            plt.savefig(network_usage_path)
            plt.close()
        else:
            network_usage_path = None

        # ------------- Visualization 5: API Usage by Trojan Category -------------
        if (
            "api_command_java.lang.runtime_exec" in available_columns
            and "api_webview_android.webkit.webview_loadurl" in available_columns
        ):
            plt.figure(figsize=(10, 6))
            filtered_df.groupby("category").agg(
                {
                    "api_command_java.lang.runtime_exec": "sum",
                    "api_webview_android.webkit.webview_loadurl": "sum",
                }
            ).plot(kind="bar", color=["orange", "skyblue"])
            plt.title(f"API Usage by Trojan Category (Filtered by {category})")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.ylabel("API Usage (Counts)")
            api_usage_path = os.path.join(
                app.config["UPLOAD_FOLDER"], "api_usage_filtered.png"
            )
            plt.savefig(api_usage_path)
            plt.close()
        else:
            api_usage_path = None

        # Pass all image paths and categories to the template
        return render_template(
            "data_visualization.html",
            categories=categories,
            selected_category=category,
            category_pie_url="/uploads/category_pie_filtered.png",
            memory_usage_url=memory_usage_path,
            battery_usage_url=battery_usage_path,
            network_usage_url=network_usage_path,
            api_usage_url=api_usage_path,
            missing_message=missing_message,  # Display the missing columns message
        )

    return redirect("/")


# Serve static files (visualization images)
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/add_user")
def add_user():
    if "role" in session and session["role"] == "admin":
        return render_template("add_user.html")
    return redirect("/")


# Route to handle creating a user
@app.route("/create_user", methods=["POST"])
def create_user():
    username = request.form["username"]
    password = request.form["password"]
    role = request.form["role"]

    # Insert user into MongoDB
    users_collection.insert_one(
        {
            "username": username,
            "password": password,  # Consider hashing passwords before storing
            "role": role,
        }
    )

    return "User created successfully!"


# Route to view all users
@app.route("/view_users", methods=["GET"])
def view_users():
    if "role" in session and session["role"] == "admin":
        users = list(users_collection.find())  # Convert cursor to a list
        return render_template("view_users.html", users=users)
    return redirect("/")


# Route to add a user (can be for AJAX requests)
@app.route("/add_users", methods=["POST"])  # Updated route
def add_users():
    data = request.get_json()
    username = data.get("username")
    role = data.get("role")
    password = data.get("password")
    if username and role:
        new_user = {"username": username, "role": role, "password": password}
        # Insert the new user into MongoDB
        users_collection.insert_one(new_user)
        return jsonify(success=True), 201
    return jsonify(success=False, message="Invalid input"), 400


@app.route("/delete_user/<user_id>", methods=["DELETE"])
def delete_user(user_id):
    # Ensure you are using ObjectId if the user_id is an ObjectId from MongoDB
    result = users_collection.delete_one({"_id": ObjectId(user_id)})
    if result.deleted_count > 0:
        return jsonify(success=True), 200
    return jsonify(success=False, message="User not found"), 404


@app.route("/update_user/<user_id>", methods=["PUT"])
def update_user(user_id):
    data = request.get_json()
    username = data.get("username")
    role = data.get("role")
    password = data.get("password")  # Include password if needed

    # Create an update dictionary
    update_data = {}
    if username:
        update_data["username"] = username
    if role:
        update_data["role"] = role
    if password:  # Only update if password is provided
        update_data["password"] = password

    result = users_collection.update_one(
        {"_id": ObjectId(user_id)}, {"$set": update_data}
    )
    if result.modified_count > 0:  # Check for modified count
        return jsonify(success=True), 200
    return jsonify(success=False, message="User not found or no update made"), 404


# Route to display the Q&A page for answering questions
@app.route("/q_and_a", methods=["GET"])
def q_and_a():
    questions = list(user_questions.find())  # Fetch all questions from MongoDB
    return render_template("q_and_a.html", questions=questions)


# Route to submit an answer to a question
@app.route("/answer_question/<question_id>", methods=["POST"])
def answer_question(question_id):
    """Submit an answer to a question."""
    data = request.json
    answer = data.get("answer")

    if not answer:
        abort(400, description="Answer is required")

    # Convert question_id to ObjectId for MongoDB lookup
    result = user_questions.update_one(
        {"_id": ObjectId(question_id)}, {"$set": {"answer": answer}}
    )

    if result.modified_count == 0:
        abort(404, description="Question not found or already answered")

    return jsonify({"success": True})

    # Route for the chatbot page


@app.route("/chatbot", methods=["GET"])
def read_chatbot():
    """Display the chatbot page."""
    return render_template("chatbot.html")

    # Route for the user chatbot page


@app.route("/chatbotuser", methods=["GET"])
def read_chatbotuser():
    """Display the chatbot page."""
    return render_template("chatbotuser.html")

    # Users Q&A Page


@app.route("/users_qa", methods=["GET"])
def users_qa():
    """Display the users' Q&A page."""
    return render_template("users_qa.html")


# Submit Question to MongoDB
@app.route("/submit_question", methods=["POST"])
def submit_question():
    """Submit a question to MongoDB."""
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"detail": "Question is required"}), 400

    # Save the question to MongoDB
    question_entry = {"question": question}
    db["user_questions"].insert_one(question_entry)

    return jsonify({"success": True})

    # Overview of Malware Page


@app.route("/overview_malware", methods=["GET"])
def overview_malware():
    """Display the overview of malware page."""
    return render_template("overview_malware.html")


# Risk Notification and Popup Page
@app.route("/risk_notification", methods=["GET"])
def risk_notification():
    """Display the risk notification page."""
    return render_template("risk_notification.html")


# Helper function to extract features from URLs
def extract_url_features(url):
    # Length of the URL
    url_length = len(url)

    # Count the number of special characters in the URL
    num_special_chars = len(re.findall(r"[?=&]", url))

    # Presence of "https" in the scheme
    has_https = 1 if urlparse(url).scheme == "https" else 0

    # Number of subdomains
    num_subdomains = urlparse(url).hostname.count(".") if urlparse(url).hostname else 0

    # Check for common phishing words
    phishing_words = ["login", "signin", "bank", "secure", "account", "update"]
    has_phishing_words = 1 if any(word in url.lower() for word in phishing_words) else 0

    return [
        url_length,
        num_special_chars,
        has_https,
        num_subdomains,
        has_phishing_words,
    ]


# Prepare column names for the features
url_feature_columns = [
    "url_length",
    "num_special_chars",
    "has_https",
    "num_subdomains",
    "has_phishing_words",
]


# Hybrid Model: Stacked Ensemble Learning for URLs
def train_hybrid_model_for_urls():
    phishing_df = pd.read_csv("phishing_urls.csv")

    # Extract features and target from the phishing dataset
    X = (
        phishing_df["URL"].apply(extract_url_features).tolist()
    )  # Feature extraction for each URL
    y = phishing_df["Label"].apply(
        lambda x: 1 if x == "bad" else 0
    )  # Label encoding: 1 for bad, 0 for good

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define base learners
    estimators = [
        ("rf", RandomForestClassifier(n_estimators=100)),
        ("gb", GradientBoostingClassifier(n_estimators=100)),
    ]

    # Define Stacking Classifier
    clf = StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression(), cv=5
    )

    # Train the model
    clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Hybrid Model Accuracy for URL classification: {accuracy * 100:.2f}%")

    return clf


# Train the model when the server starts
url_classifier = train_hybrid_model_for_urls()


@app.route("/result")
def result_page():
    if "role" in session and session["role"] == "user":
        return render_template("result.html")
    return redirect("/")


@app.route("/submit_url", methods=["POST"])
def submit_url():
    if "role" in session and session["role"] == "user":
        url = request.form["url"]

        # Extract features from the submitted URL
        url_features = np.array(extract_url_features(url)).reshape(1, -1)

        # Predict using the trained hybrid model
        prediction = url_classifier.predict(url_features)

        # Determine the result
        result = "Bad URL" if prediction[0] == 1 else "Good URL"

        # Initialize the session counters if they don't exist
        if "good_url_count" not in session:
            session["good_url_count"] = 0
        if "bad_url_count" not in session:
            session["bad_url_count"] = 0

        # Update the session counts based on the result
        if result == "Good URL":
            session["good_url_count"] += 1
        else:
            session["bad_url_count"] += 1

        # Insert the URL and result into MongoDB
        url_checks.insert_one(
            {"url": url, "result": result, "timestamp": datetime.now()}
        )

        # Keep only the latest 5 entries in MongoDB
        total_count = url_checks.count_documents({})
        if total_count > 5:
            oldest_entry = url_checks.find_one(sort=[("timestamp", 1)])
            url_checks.delete_one({"_id": oldest_entry["_id"]})

        # Send result back to the user
        return render_template("result.html", url_result=result)

    return redirect("/")


# Route to handle logout
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


# Main app start
if __name__ == "__main__":
    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        os.makedirs(app.config["UPLOAD_FOLDER"])
    app.run(debug=True)
