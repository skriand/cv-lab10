import streamlit as st
from PIL import Image
from PIL import ImageOps
from PIL import UnidentifiedImageError
import base64
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
import numpy as np
import pandas as pd
import cv2 as cv
import joblib
import math

model = joblib.load('model.sav')

def vectorize(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    _, descriptors = sift.detectAndCompute(gray, None)
    classes = model.predict(descriptors)
    hist, _ = np.histogram(classes, np.arange(129))
    return hist / hist.sum()


def get_k_neighbours(vector, df, number_of_neighbours):
    neigh = NearestNeighbors(n_neighbors=number_of_neighbours, metric=lambda a, b: distance.cosine(a, b))
    neigh.fit(df['embedding'].to_numpy().tolist())
    return neigh.kneighbors([vector], number_of_neighbours, return_distance=False)


def get_neighbours_pathes(df, neighbors):
    similar = df.iloc[neighbors[0]]
    return similar['path'].to_numpy().tolist()


st.set_option('deprecation.showfileUploaderEncoding', False)
db = pd.read_csv('data1.csv', delimiter=',')
db['embedding'] = db['embedding'].apply(lambda x: np.frombuffer(base64.b64decode(bytes(x[2:-1], encoding='ascii')), dtype=np.int32))


def main():
    st.title('Search for similar expressions')
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp", "tiff"])
    if uploaded_file is None:
        pass
    else:
        image = Image.open(uploaded_file).convert("RGB")
        image = ImageOps.exif_transpose(image)
        st.image(image)
        img_opencv = np.array(image)
        img_opencv = img_opencv[:, :, ::-1].copy()
        if  math.isnan(sum(vectorize(img_opencv))):
            st.success("Bad image")
            pass 
        else:
            pathes = get_neighbours_pathes(db, get_k_neighbours(vectorize(img_opencv), db, 3))
            st.success("Found similar images")
            col = st.columns(3)
            for i in range(len(pathes)):
                try:
                    with col[i]:
                        similar_image = Image.open(pathes[i])
                        st.image(similar_image, width=200)
                except UnidentifiedImageError:
                    pass


if __name__ == '__main__':
    main()