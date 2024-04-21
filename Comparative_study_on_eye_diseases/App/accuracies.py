import os
from model_architecture import get_model
import numpy as np
from keras.models import Model
from sklearn.svm import SVC
import joblib
import xgboost as xgb

folder_dict = {
    1: "normal",
    2: "lab_hsv",
    3: "clahecc",
    4: "gabor"
}
classes_labels = {
    0: "CNV",
    1: "DME",
    2: "DRUSEN",
    3: "NORMAL",
}


def get_model_with_weights(num):
    folder_path = f"../{folder_dict[num]}"
    weights = os.path.join(folder_path, "weights/final.weights.h5")
    model = get_model()
    model.load_weights(weights)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, folder_path


def test_cnn(model, data):
    pred_cnn = model.predict(np.expand_dims(data, axis=0))
    pred = np.argmax(pred_cnn, axis=1)
    return pred


def test_svm(data, folder_path):
    model_path = os.path.join(folder_path, "svm_model.pkl")
    model = joblib.load(model_path)
    pred = model.predict(data)
    return pred


def test_knn(data, folder_path):
    model_path = os.path.join(folder_path, "knn_model.pkl")
    model = joblib.load(model_path)
    pred = model.predict(data)
    return pred


def test_adaboost(data, folder_path):
    model_path = os.path.join(folder_path, "adaboost_model.pkl")
    model = joblib.load(model_path)
    pred = model.predict(data)
    return pred


def test_xgboost(data, folder_path):
    model_path = os.path.join(folder_path, "xgboost_model.pkl")
    dtest = xgb.DMatrix(data)
    model = joblib.load(model_path)
    pred = model.predict(dtest)
    return pred


def get_predictions(img, num):
    model, folder_path = get_model_with_weights(num)
    normalized_image = img.astype(np.float32) / 255.0

    extracted_features = Model(
        inputs=model.input, outputs=model.layers[-2].output)
    features = extracted_features.predict(
        np.expand_dims(normalized_image, axis=0))

    pred_cnn = test_cnn(model, normalized_image)
    pred_svm_poly = test_svm(features, folder_path)
    pred_knn = test_knn(features, folder_path)
    pred_adaboost = test_adaboost(features, folder_path)
    pred_xgboost = test_xgboost(features, folder_path)
    return classes_labels[pred_cnn[0]], classes_labels[pred_svm_poly[0]], classes_labels[pred_knn[0]], classes_labels[pred_adaboost[0]], classes_labels[pred_xgboost[0]]
