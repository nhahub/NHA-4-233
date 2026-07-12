from keras.applications.inception_v3 import InceptionV3
from sklearn.preprocessing import normalize

class FeatureExtractor:
    def __init__(self):
        print("loading DeepNet (Inception-V3) ...")
        self.model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
     
    def get_features(self, batch):
        features =  self.model.predict(batch)
        features = features.reshape(-1,features.shape[-1])
        return normalize(features, axis=1, norm='l2') 
