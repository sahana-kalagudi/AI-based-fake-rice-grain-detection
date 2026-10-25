from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    redirect,
    url_for
)

import os
import uuid
from datetime import datetime

import numpy as np
from PIL import Image

try:
    from tensorflow.keras.models import load_model
except Exception as e:
    load_model = None
    print("TensorFlow import failed:", e)

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.lib.enums import TA_CENTER

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)


# ============================================================
# RICEGUARD APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# FOLDERS
# ============================================================

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

REPORT_FOLDER = os.path.join(
    BASE_DIR,
    "reports"
)


os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    REPORT_FOLDER,
    exist_ok=True
)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "rice_model.keras"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 65)
print("                    RICEGUARD")
print("          AI RICE DETECTION SYSTEM")
print("=" * 65)

print()
print("Model path:")
print(MODEL_PATH)

print()
print("Model exists:")
print(os.path.exists(MODEL_PATH))


model = None


try:

    if load_model is not None and os.path.exists(MODEL_PATH):

        model = load_model(
            MODEL_PATH,
            compile=False
        )

        print()
        print("✅ CNN MODEL LOADED SUCCESSFULLY")

        print()
        print("Model input shape:")
        print(model.input_shape)

    else:

        print()
        print("❌ MODEL FILE NOT FOUND OR TENSORFLOW IS UNAVAILABLE")

        print(
            "Expected location:",
            MODEL_PATH
        )


except Exception as e:

    print()
    print("❌ MODEL LOADING ERROR")

    print(
        type(e).__name__
    )

    print(
        str(e)
    )


print("=" * 65)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        image = image.resize(
            (224, 224)
        )

        image_array = np.array(
            image,
            dtype=np.float32
        )

        image_array = (
            image_array / 255.0
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        return image_array

    except Exception as e:

        print(
            "Image preprocessing error:",
            e
        )

        return None


# ============================================================
# QUALITY GRADE
# ============================================================

def get_quality_grade(
    quality
):

    if quality >= 85:
        return "A"

    elif quality >= 70:
        return "B"

    elif quality >= 50:
        return "C"

    else:
        return "D"


# ============================================================
# CREATE PDF REPORT
# ============================================================

def create_pdf_report(
    pdf_filename,
    detection_id,
    detection_time,
    result,
    accuracy,
    quality,
    grade
):

    pdf_path = os.path.join(
        REPORT_FOLDER,
        pdf_filename
    )


    document = SimpleDocTemplate(

        pdf_path,

        pagesize=A4,

        rightMargin=50,

        leftMargin=50,

        topMargin=50,

        bottomMargin=50
    )


    styles = getSampleStyleSheet()


    # --------------------------------------------------------
    # TITLE STYLE
    # --------------------------------------------------------

    title_style = ParagraphStyle(

        "RiceGuardTitle",

        parent=styles["Title"],

        alignment=TA_CENTER,

        fontSize=24,

        leading=30,

        textColor=colors.HexColor(
            "#315425"
        ),

        spaceAfter=8
    )


    # --------------------------------------------------------
    # SUBTITLE STYLE
    # --------------------------------------------------------

    subtitle_style = ParagraphStyle(

        "RiceGuardSubtitle",

        parent=styles["Normal"],

        alignment=TA_CENTER,

        fontSize=11,

        textColor=colors.HexColor(
            "#788274"
        ),

        spaceAfter=25
    )


    # --------------------------------------------------------
    # HEADING STYLE
    # --------------------------------------------------------

    heading_style = ParagraphStyle(

        "RiceGuardHeading",

        parent=styles["Heading2"],

        fontSize=16,

        textColor=colors.HexColor(
            "#426336"
        ),

        spaceBefore=15,

        spaceAfter=12
    )


    # --------------------------------------------------------
    # NORMAL STYLE
    # --------------------------------------------------------

    normal_style = ParagraphStyle(

        "RiceGuardNormal",

        parent=styles["Normal"],

        fontSize=11,

        leading=18,

        textColor=colors.HexColor(
            "#4d5749"
        )
    )


    story = []


    # ========================================================
    # PDF HEADER
    # ========================================================

    story.append(
        Paragraph(
            "RiceGuard",
            title_style
        )
    )


    story.append(
        Paragraph(
            "AI Powered Rice Detection & Quality Analysis",
            subtitle_style
        )
    )


    story.append(
        Paragraph(
            "Rice Detection Report",
            heading_style
        )
    )


    story.append(
        Spacer(
            1,
            10
        )
    )


    # ========================================================
    # REPORT DATA
    # ========================================================

    data = [

        [
            "Detection ID",
            detection_id
        ],

        [
            "Date & Time",
            detection_time
        ],

        [
            "Prediction",
            result
        ],

        [
            "Accuracy",
            f"{accuracy:.2f}%"
        ],

        [
            "Rice Quality",
            f"{quality:.2f}%"
        ],

        [
            "Quality Grade",
            grade
        ]

    ]


    table = Table(

        data,

        colWidths=[
            160,
            300
        ]
    )


    table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor(
                    "#edf6e8"
                )
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (0, -1),
                colors.HexColor(
                    "#315425"
                )
            ),

            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),

            (
                "FONTNAME",
                (1, 0),
                (1, -1),
                "Helvetica"
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                11
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor(
                    "#d6e3d0"
                )
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                12
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                12
            )

        ])
    )


    story.append(
        table
    )


    story.append(
        Spacer(
            1,
            25
        )
    )


    # ========================================================
    # QUALITY GRADING
    # ========================================================

    story.append(
        Paragraph(
            "Quality Grading",
            heading_style
        )
    )


    grading_text = (
        "<b>A</b> : 85% and above<br/>"
        "<b>B</b> : 70% to below 85%<br/>"
        "<b>C</b> : 50% to below 70%<br/>"
        "<b>D</b> : Below 50%"
    )


    story.append(
        Paragraph(
            grading_text,
            normal_style
        )
    )


    story.append(
        Spacer(
            1,
            25
        )
    )


    # ========================================================
    # FOOTER
    # ========================================================

    story.append(
        Paragraph(
            "Generated by RiceGuard AI Rice Detection System.",
            subtitle_style
        )
    )


    document.build(
        story
    )


    return pdf_filename


# ============================================================
# HOME / UPLOAD PAGE
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def index():

    # --------------------------------------------------------
    # SHOW UPLOAD PAGE
    # --------------------------------------------------------

    if request.method == "GET":

        return render_template(
            "index.html"
        )


    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if model is None:

        return render_template(
            "index.html",
            error="CNN model could not be loaded."
        )


    # --------------------------------------------------------
    # CHECK IMAGE
    # --------------------------------------------------------

    if "image" not in request.files:

        return render_template(
            "index.html",
            error="Please select an image."
        )


    file = request.files["image"]


    if file.filename == "":

        return render_template(
            "index.html",
            error="Please select an image."
        )


    # --------------------------------------------------------
    # CHECK FILE EXTENSION
    # --------------------------------------------------------

    extension = os.path.splitext(
        file.filename
    )[1].lower()


    allowed_extensions = [

        ".jpg",

        ".jpeg",

        ".png",

        ".webp"

    ]


    if extension not in allowed_extensions:

        return render_template(

            "index.html",

            error=(
                "Please upload JPG, JPEG, PNG "
                "or WEBP image."
            )
        )


    # --------------------------------------------------------
    # UNIQUE IMAGE NAME
    # --------------------------------------------------------

    unique_name = (

        uuid.uuid4().hex

        + extension
    )


    image_full_path = os.path.join(

        UPLOAD_FOLDER,

        unique_name
    )


    # --------------------------------------------------------
    # SAVE IMAGE
    # --------------------------------------------------------

    try:

        file.save(
            image_full_path
        )

    except Exception as e:

        return render_template(

            "index.html",

            error=(
                "Unable to save image: "
                + str(e)
            )
        )


    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    image_array = preprocess_image(
        image_full_path
    )


    if image_array is None:

        return render_template(

            "index.html",

            error=(
                "Unable to process the image."
            )
        )


    # --------------------------------------------------------
    # CNN PREDICTION
    # --------------------------------------------------------

    try:

        prediction = model.predict(
            image_array,
            verbose=0
        )

        raw_value = float(
            np.asarray(
                prediction
            ).reshape(-1)[0]
        )

        print()
        print("=" * 65)

        print(
            "NEW RICE IMAGE"
        )

        print(
            "Raw CNN output:",
            raw_value
        )

        if raw_value >= 0.5:
            result = "Real Rice"
            accuracy = raw_value * 100
            quality = raw_value * 100
            print("Prediction: REAL RICE")
        else:
            result = "Fake Rice"
            fake_confidence = (1 - raw_value) * 100
            accuracy = fake_confidence
            quality = raw_value * 100
            print("Prediction: FAKE RICE")

        grade = get_quality_grade(quality)

        # ----------------------------------------------------
        # ROUND VALUES
        # ----------------------------------------------------

        accuracy = round(
            accuracy,
            2
        )

        quality = round(
            quality,
            2
        )

        # ----------------------------------------------------
        # DETECTION ID
        # ----------------------------------------------------

        detection_id = (
            "RG-"
            + datetime.now().strftime(
                "%Y%m%d%H%M%S"
            )
            + "-"
            + uuid.uuid4().hex[:6].upper()
        )

        # ----------------------------------------------------
        # DETECTION TIME
        # ----------------------------------------------------

        detection_time = (
            datetime.now().strftime(
                "%d %B %Y, %I:%M %p"
            )
        )

        # ----------------------------------------------------
        # CREATE PDF
        # ----------------------------------------------------

        report_filename = (
            detection_id
            + ".pdf"
        )

        create_pdf_report(
            report_filename,
            detection_id,
            detection_time,
            result,
            accuracy,
            quality,
            grade
        )

        print(
            "Accuracy:",
            accuracy,
            "%"
        )

        print(
            "Quality:",
            quality,
            "%"
        )

        print(
            "Grade:",
            grade
        )

        print(
            "PDF:",
            report_filename
        )

        print("=" * 65)

        # ====================================================
        # GO TO SEPARATE RESULT PAGE
        # ====================================================

        return redirect(
            url_for(
                "result_page",
                image=unique_name,
                prediction=result,
                accuracy=accuracy,
                quality=quality,
                grade=grade,
                report=report_filename,
                detection_id=detection_id,
                detection_time=detection_time
            )
        )

    except Exception as e:

        print(
            "Prediction error:",
            e
        )

        return render_template(
            "index.html",
            error=(
                "Error while analyzing image: "
                + str(e)
            )
        )


# ============================================================
# SEPARATE RESULT PAGE
# ============================================================

@app.route(
    "/result"
)
def result_page():

    image = request.args.get(
        "image"
    )

    prediction = request.args.get(
        "prediction"
    )

    accuracy = request.args.get(
        "accuracy",
        "0"
    )

    quality = request.args.get(
        "quality",
        "0"
    )

    grade = request.args.get(
        "grade",
        "D"
    )

    report = request.args.get(
        "report"
    )

    detection_id = request.args.get(
        "detection_id"
    )

    detection_time = request.args.get(
        "detection_time"
    )


    if not image or not prediction:

        return redirect(
            url_for("index")
        )


    return render_template(

        "result.html",

        image_path=(
            "uploads/"
            + image
        ),

        result=prediction,

        accuracy=float(
            accuracy
        ),

        quality=float(
            quality
        ),

        grade=grade,

        report_filename=report,

        detection_id=detection_id,

        detection_time=detection_time
    )


# ============================================================
# SEPARATE REPORT PAGE
# ============================================================

@app.route(
    "/report/<filename>"
)
def report_page(filename):

    pdf_path = os.path.join(
        REPORT_FOLDER,
        filename
    )


    if not os.path.exists(
        pdf_path
    ):

        return redirect(
            url_for("index")
        )


    return render_template(

        "report.html",

        report_filename=filename
    )


# ============================================================
# VIEW PDF IN BROWSER
# ============================================================

@app.route(
    "/view_report/<filename>"
)
def view_report(filename):

    return send_from_directory(

        REPORT_FOLDER,

        filename,

        mimetype="application/pdf",

        as_attachment=False
    )


# ============================================================
# DOWNLOAD PDF
# ============================================================

@app.route(
    "/download_report/<filename>"
)
def download_report(filename):

    return send_from_directory(

        REPORT_FOLDER,

        filename,

        mimetype="application/pdf",

        as_attachment=True
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 65)

    print(
        "              STARTING RICEGUARD"
    )

    print("=" * 65)

    print(
        "Open: http://127.0.0.1:5000"
    )

    print("=" * 65)


    app.run(
        debug=True
    )